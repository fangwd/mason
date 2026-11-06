//! Prepares the images tools return before they are stored and sent.

use std::io::Cursor;
use std::panic::AssertUnwindSafe;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use image::imageops::FilterType;
use image::{DynamicImage, ImageFormat, ImageReader, ImageResult, Limits};

use crate::llm::Image;

/// The longest side kept: Anthropic rejects larger images once a request holds
/// more than 20, and providers shrink them below this anyway.
pub const MAX_IMAGE_EDGE: u32 = 3000;
/// The largest image kept, in base64 bytes: Bedrock's per-image limit, the
/// strictest of the major providers.
pub const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;
/// The memory one decode may allocate: long screenshots fit, while a small file
/// declaring huge dimensions fails before allocating.
const MAX_DECODE_BYTES: u64 = 256 * 1024 * 1024;

/// Bounds images off the async runtime. Returns the images kept and how many
/// were dropped as unusable.
pub async fn bound_images(images: Vec<Image>) -> (Vec<Image>, usize) {
    let count = images.len();
    if count == 0 {
        return (images, 0);
    }
    let kept: Vec<Image> = tokio::task::spawn_blocking(move || {
        images
            .into_iter()
            // A decoder panic drops that image rather than the run.
            .filter_map(|image| std::panic::catch_unwind(AssertUnwindSafe(|| bound(image))).ok()?)
            .collect()
    })
    .await
    .unwrap_or_default();
    let dropped = count - kept.len();
    (kept, dropped)
}

/// The image labelled by its actual format and shrunk within [`MAX_IMAGE_EDGE`]
/// and [`MAX_IMAGE_BYTES`], or `None` unless it is a PNG, JPEG, WebP or GIF,
/// the formats every major provider accepts.
pub fn bound(image: Image) -> Option<Image> {
    fit(image, MAX_IMAGE_EDGE, MAX_IMAGE_BYTES)
}

fn fit(image: Image, max_edge: u32, max_bytes: usize) -> Option<Image> {
    let bytes = STANDARD.decode(&image.data).ok()?;
    let reader = ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .ok()?;
    let mut format = reader.format().filter(|format| {
        matches!(
            format,
            ImageFormat::Png | ImageFormat::Jpeg | ImageFormat::WebP | ImageFormat::Gif
        )
    })?;
    let (width, height) = reader.into_dimensions().ok()?;
    if width.max(height) <= max_edge && image.data.len() <= max_bytes {
        return Some(Image {
            mime_type: format.to_mime_type().to_owned(),
            data: image.data,
        });
    }
    let mut decoded = decode(&bytes, format).ok()?;
    if width.max(height) > max_edge {
        decoded = decoded.resize(max_edge, max_edge, FilterType::CatmullRom);
    }
    loop {
        let data = encode(&decoded, format)?;
        if data.len() <= max_bytes {
            return Some(Image {
                mime_type: format.to_mime_type().to_owned(),
                data,
            });
        }
        if format == ImageFormat::Jpeg {
            decoded = decoded.resize(
                decoded.width() * 3 / 4,
                decoded.height() * 3 / 4,
                FilterType::CatmullRom,
            );
        } else {
            // JPEG is far smaller for photo-like content.
            format = ImageFormat::Jpeg;
            decoded = DynamicImage::ImageRgb8(decoded.to_rgb8());
        }
    }
}

fn decode(bytes: &[u8], format: ImageFormat) -> ImageResult<DynamicImage> {
    let mut reader = ImageReader::with_format(Cursor::new(bytes), format);
    let mut limits = Limits::default();
    limits.max_alloc = Some(MAX_DECODE_BYTES);
    reader.limits(limits);
    reader.decode()
}

fn encode(image: &DynamicImage, format: ImageFormat) -> Option<String> {
    let mut encoded = Cursor::new(Vec::new());
    image.write_to(&mut encoded, format).ok()?;
    Some(STANDARD.encode(encoded.into_inner()))
}

#[cfg(test)]
mod tests {
    use image::{GenericImageView, RgbImage};

    use super::*;

    fn encoded(image: RgbImage, format: ImageFormat) -> Image {
        Image {
            mime_type: format.to_mime_type().to_owned(),
            data: encode(&DynamicImage::ImageRgb8(image), format).unwrap(),
        }
    }

    fn noise(width: u32, height: u32) -> RgbImage {
        let mut state = 1u32;
        RgbImage::from_fn(width, height, |_, _| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let [r, g, b, _] = state.to_le_bytes();
            image::Rgb([r, g, b])
        })
    }

    fn dimensions(image: &Image) -> (u32, u32) {
        let bytes = STANDARD.decode(&image.data).unwrap();
        ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .unwrap()
            .into_dimensions()
            .unwrap()
    }

    #[test]
    fn oversized_images_shrink_to_the_edge_cap_in_their_own_format() {
        for format in [ImageFormat::Png, ImageFormat::Jpeg] {
            let tall = bound(encoded(RgbImage::new(400, 6000), format)).unwrap();
            assert_eq!(dimensions(&tall), (200, 3000));
            assert_eq!(tall.mime_type, format.to_mime_type());
        }
    }

    #[test]
    fn heavy_images_are_reencoded_within_the_byte_cap() {
        let heavy = encoded(noise(300, 300), ImageFormat::Png);
        let max_bytes = heavy.data.len() / 4;
        let fitted = fit(heavy, MAX_IMAGE_EDGE, max_bytes).unwrap();
        assert!(fitted.data.len() <= max_bytes);
        assert_eq!(fitted.mime_type, "image/jpeg");
    }

    #[test]
    fn images_within_the_caps_pass_unchanged_but_labelled_by_their_bytes() {
        let png = encoded(RgbImage::new(3000, 100), ImageFormat::Png);
        let mislabelled = Image {
            mime_type: "image/jpeg".into(),
            ..png.clone()
        };
        assert_eq!(bound(mislabelled), Some(png));
    }

    #[test]
    fn unsupported_or_undecodable_images_are_dropped() {
        let svg = Image {
            mime_type: "image/svg+xml".into(),
            data: STANDARD.encode(br#"<svg xmlns="http://www.w3.org/2000/svg"/>"#),
        };
        let garbage = Image {
            mime_type: "image/png".into(),
            data: "not base64!".into(),
        };
        assert_eq!(bound(svg), None);
        assert_eq!(bound(garbage), None);
    }

    /// A PNG whose header declares `width` x `height` RGB pixels but holds none.
    fn png_header(width: u32, height: u32) -> Vec<u8> {
        fn crc(bytes: &[u8]) -> u32 {
            !bytes.iter().fold(!0u32, |crc, byte| {
                (0..8).fold(crc ^ u32::from(*byte), |crc, _| {
                    (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg())
                })
            })
        }
        let mut chunk = b"IHDR".to_vec();
        chunk.extend(width.to_be_bytes());
        chunk.extend(height.to_be_bytes());
        chunk.extend([8, 2, 0, 0, 0]);
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend(13u32.to_be_bytes());
        png.extend(&chunk);
        png.extend(crc(&chunk).to_be_bytes());
        // An empty zlib stream: the decoder reads up to the image data first.
        let idat = b"IDAT\x78\x9c\x03\x00\x00\x00\x00\x01";
        png.extend(8u32.to_be_bytes());
        png.extend(idat);
        png.extend(crc(idat).to_be_bytes());
        png.extend(0u32.to_be_bytes());
        png.extend(b"IEND");
        png.extend(crc(b"IEND").to_be_bytes());
        png
    }

    #[test]
    fn huge_declared_dimensions_fail_before_decoding() {
        // 300 MB of RGB: under the crate's default limit, over ours.
        let bomb = png_header(10_000, 10_000);
        let result = decode(&bomb, ImageFormat::Png);
        assert!(
            matches!(result, Err(image::ImageError::Limits(_))),
            "{:?}",
            result.map(|image| image.dimensions())
        );
        let image = Image {
            mime_type: "image/png".into(),
            data: STANDARD.encode(&bomb),
        };
        assert_eq!(bound(image), None);
    }

    #[test]
    fn gifs_are_kept() {
        let gif = encoded(RgbImage::new(4, 4), ImageFormat::Gif);
        assert_eq!(bound(gif.clone()), Some(gif));
    }

    #[tokio::test]
    async fn bound_images_reports_the_dropped_count() {
        let png = encoded(RgbImage::new(4, 4), ImageFormat::Png);
        let svg = Image {
            mime_type: "image/svg+xml".into(),
            data: STANDARD.encode(b"<svg/>"),
        };
        let (kept, dropped) = bound_images(vec![png.clone(), svg]).await;
        assert_eq!((kept, dropped), (vec![png], 1));
    }
}
