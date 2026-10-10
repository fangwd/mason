//! CLI one-shot run tests against a mock vendor.

use tokio::process::Command;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const HELLO_SSE: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hello \"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]} \n\n",
    "data: [DONE]\n\n",
);

fn mason_run() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mason"));
    command.args(["run", "--execution-mode", "trusted", "hi"]);
    command
}

fn vendor_env<'a>(command: &'a mut Command, server: &MockServer) -> &'a mut Command {
    command
        .env("AI_VENDORS", "mock")
        .env("AI_MOCK_URL", server.uri())
        .env("AI_MOCK_MODEL", "mock-model")
        .env("AI_MOCK_API_KEY", "mock-key")
}

#[tokio::test]
async fn run_streams_the_answer_and_exits_zero() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(HELLO_SSE),
        )
        .expect(1)
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let output = vendor_env(&mut mason_run(), &server)
        .current_dir(dir.path())
        .arg("--cwd")
        .arg(dir.path())
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Hello world"), "streamed answer: {stdout}");
}

#[tokio::test]
async fn a_closed_stdout_does_not_panic_the_run() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(HELLO_SSE),
        )
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let mut child = vendor_env(&mut mason_run(), &server)
        .current_dir(dir.path())
        .arg("--cwd")
        .arg(dir.path())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    // `mason run ... | head -1`: the reader goes away before the answer streams.
    drop(child.stdout.take());
    let output = child.wait_with_output().await.unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert_eq!(output.status.code(), Some(0), "{stderr}");
}

#[tokio::test]
async fn a_provider_failure_exits_one_with_a_message_on_stderr() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .expect(1)
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let output = vendor_env(&mut mason_run(), &server)
        .current_dir(dir.path())
        .arg("--cwd")
        .arg(dir.path())
        .output()
        .await
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("HTTP 401"), "{stderr}");
    assert!(stderr.contains("unauthorized"), "{stderr}");
}

#[tokio::test]
async fn missing_vendor_configuration_exits_one() {
    let dir = tempfile::tempdir().unwrap();
    let output = mason_run()
        .env_clear()
        .current_dir(dir.path())
        .arg("--cwd")
        .arg(dir.path())
        .output()
        .await
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("AI_VENDORS"), "{stderr}");
}

#[tokio::test]
async fn run_prints_the_complete_answer_despite_a_lossy_queue() {
    // A 5,000-character answer delivered as one-character deltas: far more
    // events than the printer queue holds, so the engine drops some.
    const CHUNKS: usize = 5_000;
    let answer: String = (0..CHUNKS)
        .map(|index| char::from(b'a' + (index % 26) as u8))
        .collect();
    let mut body = String::new();
    for character in answer.chars() {
        body.push_str(&format!(
            "data: {}\n\n",
            serde_json::json!({"choices": [{"delta": {"content": character.to_string()}}]})
        ));
    }
    body.push_str("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n");
    body.push_str("data: [DONE]\n\n");

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .expect(1)
        .mount(&server)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let output = vendor_env(&mut mason_run(), &server)
        .current_dir(dir.path())
        .arg("--cwd")
        .arg(dir.path())
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        stdout,
        format!("{answer}\n"),
        "stdout holds the complete answer ({} chars printed)",
        stdout.len()
    );
}

/// The built `mcp_fixture` example (etc/check.sh builds it before tests).
#[cfg(feature = "mcp")]
fn fixture_path() -> String {
    let target = std::env::var("CARGO_TARGET_DIR")
        .unwrap_or_else(|_| format!("{}/target", env!("CARGO_MANIFEST_DIR")));
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    format!("{target}/{profile}/examples/mcp_fixture")
}

fn tool_call_sse(name: &str, arguments: &str) -> String {
    format!(
        "data: {}\n\ndata: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\ndata: [DONE]\n\n",
        serde_json::json!({"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "c1",
            "function": {"name": name, "arguments": arguments}
        }]}}]})
    )
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[tokio::test]
async fn cli_sandbox_runs_file_tools_and_blocks_host_reads_and_writes() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(workspace.join("sub")).unwrap();
    std::fs::create_dir_all(workspace.join(".mason")).unwrap();
    std::fs::write(workspace.join("inside"), "workspace content").unwrap();
    std::fs::write(workspace.join(".mason/mason.db"), "session secret").unwrap();
    let outside = dir.path().join("outside");
    std::fs::write(&outside, "private host content").unwrap();
    let sub = workspace.canonicalize().unwrap().join("sub");

    for (tool, arguments, expected) in [
        (
            "read",
            serde_json::json!({"filePath":"inside"}),
            "workspace content".to_owned(),
        ),
        (
            "bash",
            serde_json::json!({"command":format!("cat '{}'", outside.display())}),
            "[exit code 1]".to_owned(),
        ),
        (
            "bash",
            serde_json::json!({"command":format!("echo overwrite > '{}'", outside.display())}),
            "[exit code 1]".to_owned(),
        ),
        (
            "read",
            serde_json::json!({"filePath":"../outside"}),
            "permission denied".to_owned(),
        ),
        (
            "bash",
            serde_json::json!({"command":"pwd","workdir":"sub"}),
            format!("{}\n", sub.display()),
        ),
        (
            "bash",
            serde_json::json!({"command":"cat .mason/mason.db"}),
            "[exit code 1]".to_owned(),
        ),
        (
            "read",
            serde_json::json!({"filePath":".mason/mason.db"}),
            "Error".to_owned(),
        ),
    ] {
        let server = MockServer::start().await;
        mount_tool_then_answer(&server, tool, &arguments.to_string(), "done").await;
        let mut command = Command::new(env!("CARGO_BIN_EXE_mason"));
        command
            .args(["run", "--execution-mode=sandbox", "inspect"])
            .current_dir(&workspace);
        let output = vendor_env(&mut command, &server).output().await.unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let requests = server.received_requests().await.unwrap();
        let second: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
        let tool_output = second["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message["role"] == "tool")
            .unwrap()["content"]
            .as_str()
            .unwrap();
        assert!(
            tool_output.contains(&expected),
            "{tool} {arguments}: {tool_output}"
        );
        assert!(!tool_output.contains("private host content"));
        assert!(!tool_output.contains("session secret"), "{tool_output}");
        assert_eq!(
            std::fs::read_to_string(&outside).unwrap(),
            "private host content"
        );
    }
}

/// Two edits of one file in a single reply run in separate sandbox workers;
/// both must land.
#[cfg(any(target_os = "macos", target_os = "linux"))]
#[tokio::test]
async fn concurrent_sandboxed_edits_of_one_file_are_serialized() {
    let dir = tempfile::tempdir().unwrap();
    let lines: String = (0..40).map(|i| format!("line {i}\n")).collect();
    std::fs::write(dir.path().join("target.txt"), &lines).unwrap();
    let calls: Vec<serde_json::Value> = (0..40)
        .map(|i| {
            serde_json::json!({
                "index": i,
                "id": format!("c{i}"),
                "function": {
                    "name": "edit",
                    "arguments": serde_json::json!({
                        "filePath": "target.txt",
                        "oldString": format!("line {i}\n"),
                        "newString": format!("edited {i}\n"),
                    }).to_string(),
                }
            })
        })
        .collect();
    let body = format!(
        "data: {}\n\ndata: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"tool_calls\"}}]}}\n\ndata: [DONE]\n\n",
        serde_json::json!({"choices": [{"delta": {"tool_calls": calls}}]})
    );
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(body),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(HELLO_SSE),
        )
        .mount(&server)
        .await;
    let mut command = Command::new(env!("CARGO_BIN_EXE_mason"));
    command
        .args([
            "run",
            "--execution-mode=sandbox",
            "--max-total-tool-calls",
            "100",
            "inspect",
        ])
        .current_dir(dir.path());
    let output = vendor_env(&mut command, &server).output().await.unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Only 16 calls per reply run; each of them must have landed.
    let edited = std::fs::read_to_string(dir.path().join("target.txt")).unwrap();
    for i in 0..16 {
        assert!(edited.contains(&format!("edited {i}\n")), "{edited}");
    }
}

/// Background processes outlive the bash call that started them on both
/// platforms (bwrap's pid namespace must not be torn down with the shell).
#[cfg(any(target_os = "macos", target_os = "linux"))]
#[tokio::test]
async fn sandboxed_background_processes_survive_between_calls() {
    let dir = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    for command in [
        "(sleep 1; echo survived > marker) & echo started",
        "sleep 2; cat marker",
    ] {
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(tool_call_sse(
                        "bash",
                        &serde_json::json!({"command": command}).to_string(),
                    )),
            )
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
    }
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(HELLO_SSE),
        )
        .expect(1)
        .mount(&server)
        .await;
    let mut command = Command::new(env!("CARGO_BIN_EXE_mason"));
    command
        .args(["run", "--execution-mode=sandbox", "inspect"])
        .current_dir(dir.path());
    let output = vendor_env(&mut command, &server).output().await.unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = server.received_requests().await.unwrap();
    let last: serde_json::Value = serde_json::from_slice(&requests[2].body).unwrap();
    let outputs: Vec<&str> = last["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|message| message["role"] == "tool")
        .map(|message| message["content"].as_str().unwrap())
        .collect();
    assert!(outputs[0].contains("started"), "{}", outputs[0]);
    assert!(outputs[1].contains("survived"), "{}", outputs[1]);
}

#[tokio::test]
async fn serve_requires_a_token_before_starting_any_resources() {
    let dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_mason"))
        .args(["serve", "--listen", "0.0.0.0:0"])
        .env_clear()
        .current_dir(dir.path())
        .output()
        .await
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("MASON_API_TOKEN"));
    assert!(!dir.path().join("data").exists());
}

#[tokio::test]
async fn api_token_values_are_hidden_in_help() {
    let output = Command::new(env!("CARGO_BIN_EXE_mason"))
        .args(["serve", "--help"])
        .env("MASON_API_TOKEN", "private-secret-token-for-help")
        .output()
        .await
        .unwrap();
    assert!(output.status.success());
    assert!(!String::from_utf8_lossy(&output.stdout).contains("private-secret-token-for-help"));
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[tokio::test]
async fn sandbox_network_requires_a_grant_and_does_not_inherit_provider_keys() {
    for network in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/probe"))
            .respond_with(ResponseTemplate::new(200).set_body_string("network granted"))
            .expect(u64::from(network))
            .mount(&server)
            .await;
        let arguments = serde_json::json!({"command": format!("printenv PRIVATE_PROVIDER_KEY; curl --max-time 2 --silent --show-error {}/probe", server.uri())});
        mount_tool_then_answer(&server, "bash", &arguments.to_string(), "done").await;
        let mut command = Command::new(env!("CARGO_BIN_EXE_mason"));
        command
            .args(["run", "--execution-mode=sandbox", "inspect"])
            .current_dir(dir.path());
        if network {
            command.arg("--allow-network");
        }
        let output = vendor_env(&mut command, &server)
            .env("PRIVATE_PROVIDER_KEY", "sensitive-provider-key")
            .output()
            .await
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let requests = server.received_requests().await.unwrap();
        let last = requests
            .iter()
            .rev()
            .find(|request| request.url.path() == "/chat/completions")
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&last.body).unwrap();
        let tool_output = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|message| message["role"] == "tool")
            .unwrap()["content"]
            .as_str()
            .unwrap();
        assert!(
            !tool_output.contains("sensitive-provider-key"),
            "{tool_output}"
        );
        assert_eq!(
            tool_output.contains("network granted"),
            network,
            "{tool_output}"
        );
    }
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[tokio::test]
async fn sandboxed_shells_keep_developer_tools_std_streams_and_proxy_settings() {
    let dir = tempfile::tempdir().unwrap();
    let server = MockServer::start().await;
    let arguments = serde_json::json!({"command": "git --version >/dev/null && echo git-ok; echo err-ok > /dev/stderr; cat <(echo fd-ok); echo \"proxy=$HTTPS_PROXY\""});
    mount_tool_then_answer(&server, "bash", &arguments.to_string(), "done").await;
    let mut command = Command::new(env!("CARGO_BIN_EXE_mason"));
    command
        .args(["run", "--execution-mode=sandbox", "inspect"])
        .current_dir(dir.path());
    let output = vendor_env(&mut command, &server)
        .env("HTTPS_PROXY", "http://proxy.internal:3128")
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = server.received_requests().await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    let tool_output = body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["role"] == "tool")
        .unwrap()["content"]
        .as_str()
        .unwrap();
    for expected in [
        "git-ok",
        "err-ok",
        "fd-ok",
        "proxy=http://proxy.internal:3128",
    ] {
        assert!(tool_output.contains(expected), "{expected}: {tool_output}");
    }
}

/// Mounts a tool-call turn followed by an answer turn.
async fn mount_tool_then_answer(server: &MockServer, name: &str, arguments: &str, answer: &str) {
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(tool_call_sse(name, arguments)),
        )
        .up_to_n_times(1)
        .expect(1)
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(format!(
                    "data: {}\n\ndata: {{\"choices\":[{{\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n",
                    serde_json::json!({"choices": [{"delta": {"content": answer}}]})
                )),
        )
        .expect(1)
        .mount(server)
        .await;
}

#[tokio::test]
async fn run_mode_loads_skills() {
    let server = MockServer::start().await;
    mount_tool_then_answer(&server, "skill", r#"{"name":"deploy"}"#, "skill loaded").await;

    let dir = tempfile::tempdir().unwrap();
    let skill = dir.path().join("skills/deploy");
    std::fs::create_dir_all(&skill).unwrap();
    std::fs::write(
        skill.join("SKILL.md"),
        "---\nname: deploy\ndescription: Ships it\n---\nStep one.\n",
    )
    .unwrap();

    let output = vendor_env(&mut mason_run(), &server)
        .current_dir(dir.path())
        .args([
            "--cwd",
            dir.path().to_str().unwrap(),
            "--skills-dir",
            skill.parent().unwrap().to_str().unwrap(),
        ])
        .arg("deploy it")
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("skill loaded"));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("> skill"),
        "tool activity on stderr"
    );
}

#[tokio::test]
async fn run_mode_selects_a_profile() {
    let server = MockServer::start().await;
    mount_tool_then_answer(&server, "grep", r#"{"pattern":"needle"}"#, "found needle").await;

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.txt"), "needle here\n").unwrap();

    let output = vendor_env(&mut mason_run(), &server)
        .current_dir(dir.path())
        .args(["--cwd", dir.path().to_str().unwrap(), "--profile", "build"])
        .arg("find needle")
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("found needle"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("> grep"));

    // Subagent profiles cannot run top-level.
    let output = vendor_env(&mut mason_run(), &server)
        .current_dir(dir.path())
        .args([
            "--cwd",
            dir.path().to_str().unwrap(),
            "--profile",
            "explore",
        ])
        .arg("find needle")
        .output()
        .await
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("subagent profile"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(feature = "mcp")]
#[tokio::test]
async fn run_mode_connects_mcp_servers() {
    let server = MockServer::start().await;
    mount_tool_then_answer(
        &server,
        "fixture_echo",
        r#"{"text":"mcp hi"}"#,
        "echoed via mcp",
    )
    .await;

    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("mcp.json");
    std::fs::write(
        &config,
        serde_json::json!({
            "fixture": {"type": "local", "command": [fixture_path()], "timeout": 20000}
        })
        .to_string(),
    )
    .unwrap();

    let output = vendor_env(&mut mason_run(), &server)
        .current_dir(dir.path())
        .args([
            "--cwd",
            dir.path().to_str().unwrap(),
            "--allow-mcp",
            "--mcp",
            config.to_str().unwrap(),
        ])
        .arg("echo something")
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("echoed via mcp"));
    assert!(String::from_utf8_lossy(&output.stderr).contains("> fixture_echo"));
}

const VERDICT_SSE: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"v1\",\"type\":\"function\",\"function\":{\"name\":\"verdict\",\"arguments\":\"{\\\"outcome\\\":\\\"partial\\\",\\\"reason\\\":\\\"only greeted\\\"}\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test]
async fn run_reports_the_evaluator_verdict() {
    let main = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(HELLO_SSE),
        )
        .expect(1)
        .mount(&main)
        .await;
    let judge = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(VERDICT_SSE),
        )
        .expect(1)
        .mount(&judge)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let output = vendor_env(&mut mason_run(), &main)
        .env("AI_EVALUATOR_VENDORS", "judge")
        .env("AI_JUDGE_URL", judge.uri())
        .env("AI_JUDGE_MODEL", "cheap-model")
        .current_dir(dir.path())
        .arg("--cwd")
        .arg(dir.path())
        .output()
        .await
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "stderr: {stderr}");
    assert!(
        stderr.contains("> evaluated: partial (only greeted)"),
        "stderr: {stderr}"
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Hello world"));
}
