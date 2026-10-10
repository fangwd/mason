//! Subagent delegation, session lifecycle, and snapshot tests.

mod support;

use std::sync::Arc;
use std::time::Duration;

use mason::{Agent, CancellationToken, OutputSpill, Profile, PromptSource, builtin_tools};
use support::EchoTool;
use support::{Reply, ScriptedModel, answer, call, calls, is_summary_request};

fn agent(dir: &std::path::Path, model: Arc<ScriptedModel>) -> mason::AgentBuilder {
    Agent::builder()
        .model(model)
        .cwd(dir)
        .tools(builtin_tools())
        .spill(Arc::new(OutputSpill::new(dir.join("spill"))))
}

fn skill_fixture(dir: &std::path::Path) {
    let skill = dir.join("skills/deploy");
    std::fs::create_dir_all(&skill).unwrap();
    std::fs::write(
        skill.join("SKILL.md"),
        "---\nname: deploy\ndescription: Ships it carefully\n---\nStep one.\n",
    )
    .unwrap();
}

fn task_call(agent_name: &str, prompt: &str) -> Reply {
    call(
        "task",
        &format!(r#"{{"description":"d","prompt":"{prompt}","agent":"{agent_name}"}}"#),
    )
}

#[tokio::test]
async fn a_subagent_runs_in_an_independent_conversation() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        task_call("general", "investigate the thing"),
        answer("child done"),
        answer("parent done"),
    ]);
    let built = agent(dir.path(), model.clone()).build().unwrap();
    let outcome = built.run_prompt("go", None).await.unwrap();
    assert_eq!(outcome.answer, "parent done");

    let requests = model.requests();
    assert_eq!(requests.len(), 3);

    // The parent saw the task tool listing its permitted children.
    let parent_tools: Vec<&str> = requests[0]
        .tools
        .iter()
        .map(|spec| spec.function.name.as_str())
        .collect();
    assert!(parent_tools.contains(&"task"), "{parent_tools:?}");
    let task_spec = requests[0]
        .tools
        .iter()
        .find(|spec| spec.function.name == "task")
        .unwrap();
    assert!(
        task_spec.function.description.contains("- general:"),
        "{task_spec:?}"
    );

    // The child got a fresh conversation: system prompt + the task prompt,
    // and no task tool of its own (children cannot delegate).
    let child = &requests[1];
    assert_eq!(child.messages.len(), 2);
    assert!(
        child.messages[0]
            .content
            .as_deref()
            .is_some_and(|s| s.contains("You are an agent")),
        "{child:?}"
    );
    assert_eq!(
        child.messages[1].content.as_deref(),
        Some("investigate the thing")
    );
    let child_tools: Vec<&str> = child
        .tools
        .iter()
        .map(|spec| spec.function.name.as_str())
        .collect();
    assert!(!child_tools.contains(&"task"), "{child_tools:?}");

    // The parent received the wrapped answer.
    let tool_result = outcome
        .messages
        .iter()
        .find(|message| message.role == mason::Role::Tool)
        .unwrap();
    let content = tool_result.content.clone().unwrap();
    assert!(content.contains(r#"<task agent="general">"#), "{content}");
    assert!(
        content.contains("<task_result>\nchild done\n</task_result>"),
        "{content}"
    );
}

#[tokio::test]
async fn only_permitted_subagents_launch() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![task_call("build", "sneaky"), answer("done")]);
    let built = agent(dir.path(), model.clone()).build().unwrap();
    let outcome = built.run_prompt("go", None).await.unwrap();
    let tool_result = outcome
        .messages
        .iter()
        .find(|message| message.role == mason::Role::Tool)
        .unwrap()
        .content
        .clone()
        .unwrap();
    assert!(
        tool_result.contains("\"build\" is not a permitted subagent"),
        "{tool_result}"
    );
    assert!(tool_result.contains("Permitted: general"), "{tool_result}");
    // Only the parent's requests happened; no child ran.
    assert_eq!(model.request_count(), 2);
}

#[tokio::test]
async fn domain_profiles_list_only_their_permitted_children() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        task_call("explore", "find things"),
        answer("child ok"),
        answer("done"),
    ]);
    let restricted = Profile {
        allowed_subagents: vec!["explore".into()],
        ..Profile::new("research", "research-only agent")
    };
    let built = agent(dir.path(), model.clone())
        .profiles([restricted])
        .build()
        .unwrap();
    let outcome = built.run_prompt_as("research", "go", None).await.unwrap();
    assert_eq!(outcome.answer, "done");

    let requests = model.requests();
    let task_spec = requests[0]
        .tools
        .iter()
        .find(|spec| spec.function.name == "task")
        .expect("the profile allows task with a nonempty allowlist");
    assert!(
        task_spec.function.description.contains("- explore:"),
        "{task_spec:?}"
    );
    assert!(
        !task_spec.function.description.contains("- general:"),
        "{task_spec:?}"
    );
    // The child ran with the explore profile: read-only tools only.
    let child_tools: Vec<&str> = requests[1]
        .tools
        .iter()
        .map(|spec| spec.function.name.as_str())
        .collect();
    assert_eq!(child_tools, ["read", "grep", "glob"], "{child_tools:?}");
}

#[tokio::test(start_paused = true)]
async fn subagents_in_one_reply_run_concurrently_under_the_parent_token() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        calls(vec![
            (
                "t1".into(),
                "task".into(),
                r#"{"description":"a","prompt":"first","agent":"general"}"#.into(),
            ),
            (
                "t2".into(),
                "task".into(),
                r#"{"description":"b","prompt":"second","agent":"general"}"#.into(),
            ),
        ]),
        Reply::After(Duration::from_millis(100), Box::new(answer("c1"))),
        Reply::After(Duration::from_millis(100), Box::new(answer("c2"))),
        answer("done"),
    ]);
    let built = agent(dir.path(), model).build().unwrap();
    let started = tokio::time::Instant::now();
    let outcome = built.run_prompt("go", None).await.unwrap();
    assert_eq!(outcome.answer, "done");
    let results: Vec<String> = outcome
        .messages
        .iter()
        .filter(|message| message.role == mason::Role::Tool)
        .map(|message| message.content.clone().unwrap())
        .collect();
    assert_eq!(results.len(), 2);
    assert!(results.iter().any(|r| r.contains("c1")) && results.iter().any(|r| r.contains("c2")));
    assert!(
        started.elapsed() < Duration::from_millis(180),
        "the children overlapped: {:?}",
        started.elapsed()
    );
}

#[tokio::test(start_paused = true)]
async fn cancelling_the_parent_cancels_the_children() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![task_call("general", "slow child"), Reply::Hang]);
    let built = agent(dir.path(), model).build().unwrap();
    let token = CancellationToken::new();
    let canceller = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        canceller.cancel();
    });
    let err = built
        .run_prompt_as_with_token("build", "go", None, token)
        .await
        .unwrap_err();
    assert!(matches!(err, mason::Error::Cancelled), "{err}");
}

#[tokio::test]
async fn restricted_variants_advertise_and_enforce_their_own_arguments() {
    use schemars::JsonSchema;
    use serde::Deserialize;

    #[derive(Deserialize, JsonSchema)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct ReadOnlyFetchArgs {
        /// The URL to fetch.
        url: String,
    }

    struct ReadOnlyFetch;

    #[async_trait::async_trait]
    impl mason::TypedTool for ReadOnlyFetch {
        type Args = ReadOnlyFetchArgs;
        fn name(&self) -> &str {
            "webfetch_ro"
        }
        fn description(&self) -> &str {
            "Fetch a URL, read-only."
        }
        async fn run(
            &self,
            args: ReadOnlyFetchArgs,
            _ctx: &mason::ToolCtx,
        ) -> Result<String, mason::ToolError> {
            Ok(format!("fetched {}", args.url))
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![
        calls(vec![
            (
                "ok".into(),
                "webfetch_ro".into(),
                r#"{"url":"https://x"}"#.into(),
            ),
            (
                "bad".into(),
                "webfetch_ro".into(),
                r#"{"url":"https://x","saveTo":"/tmp/pwn"}"#.into(),
            ),
        ]),
        answer("done"),
    ]);
    let profile = Profile {
        tools: Some(vec!["webfetch_ro".into()]),
        ..Profile::new("triage-like", "read-only investigator")
    };
    let built = agent(dir.path(), model.clone())
        .tool(Box::new(mason::Typed(ReadOnlyFetch)))
        .profiles([profile])
        .build()
        .unwrap();
    let outcome = built
        .run_prompt_as("triage-like", "go", None)
        .await
        .unwrap();

    // The spec advertises only the permitted argument.
    let spec = &model.requests()[0].tools[0];
    assert_eq!(spec.function.name, "webfetch_ro");
    let properties = spec.function.parameters["properties"].as_object().unwrap();
    assert_eq!(
        properties.len(),
        1,
        "only `url` is advertised: {properties:?}"
    );

    let results: Vec<String> = outcome
        .messages
        .iter()
        .filter(|message| message.role == mason::Role::Tool)
        .map(|message| message.content.clone().unwrap())
        .collect();
    assert_eq!(results[0], "fetched https://x");
    assert!(results[1].contains("unknown field `saveTo`"), "{results:?}");
}

#[tokio::test]
async fn skills_are_listed_in_the_prompt_and_loaded_with_the_skill_tool() {
    let dir = tempfile::tempdir().unwrap();
    skill_fixture(dir.path());
    let model = ScriptedModel::new(vec![call("skill", r#"{"name":"deploy"}"#), answer("done")]);
    let built = agent(dir.path(), model.clone())
        .skills_dir(dir.path().join("skills"))
        .build()
        .unwrap();
    let outcome = built.run_prompt("go", None).await.unwrap();
    assert_eq!(outcome.answer, "done");

    let request = &model.requests()[0];
    let system = request.messages[0].content.clone().unwrap();
    assert!(system.contains("<available_skills>"), "{system}");
    assert!(system.contains("<name>deploy</name>"), "{system}");
    assert!(
        system.contains("<description>Ships it carefully</description>"),
        "{system}"
    );
    let tool_names: Vec<&str> = request
        .tools
        .iter()
        .map(|spec| spec.function.name.as_str())
        .collect();
    assert!(tool_names.contains(&"skill"), "{tool_names:?}");

    let loaded = outcome
        .messages
        .iter()
        .find(|message| message.role == mason::Role::Tool)
        .unwrap()
        .content
        .clone()
        .unwrap();
    assert!(
        loaded.contains(r#"<skill_content name="deploy">"#),
        "{loaded}"
    );
    assert!(loaded.contains("Step one."), "{loaded}");
    assert!(
        loaded.contains("Base directory for this skill:"),
        "{loaded}"
    );
}

#[tokio::test]
async fn prompt_reloads_update_task_and_skill_descriptions_for_new_runs() {
    let dir = tempfile::tempdir().unwrap();
    skill_fixture(dir.path());
    let path = dir.path().join("prompts.yaml");
    std::fs::write(&path, mason::prompt::EMBEDDED_PROMPTS).unwrap();

    let model = ScriptedModel::new(vec![answer("one"), answer("two")]);
    let built = agent(dir.path(), model.clone())
        .skills_dir(dir.path().join("skills"))
        .prompts(PromptSource::File(path.clone()))
        .build()
        .unwrap();

    built.run_once("first").await.unwrap();
    // Reload: a new run picks the changed descriptions up.
    let updated = std::fs::read_to_string(&path)
        .unwrap()
        .replacen(
            "Launch a new agent to handle complex, multistep tasks autonomously.",
            "RELOADED task description.",
            1,
        )
        .replacen(
            "Load a specialized skill when the task at hand matches one of the skills listed in the system prompt.",
            "RELOADED skill description.",
            1,
        );
    std::fs::write(&path, updated).unwrap();
    built.run_once("second").await.unwrap();

    let requests = model.requests();
    assert_eq!(requests.len(), 2);
    let task_desc = |request: &mason::ChatRequest| {
        request
            .tools
            .iter()
            .find(|spec| spec.function.name == "task")
            .map(|spec| spec.function.description.clone())
            .unwrap_or_default()
    };
    let skill_desc = |request: &mason::ChatRequest| {
        request
            .tools
            .iter()
            .find(|spec| spec.function.name == "skill")
            .map(|spec| spec.function.description.clone())
            .unwrap_or_default()
    };
    assert!(
        task_desc(&requests[0]).starts_with("Launch a new agent"),
        "{requests:?}"
    );
    assert!(
        task_desc(&requests[1]).starts_with("RELOADED task description."),
        "{requests:?}"
    );
    assert!(skill_desc(&requests[0]).starts_with("Load a specialized skill"));
    assert!(skill_desc(&requests[1]).starts_with("RELOADED skill description."));
    // A malformed reload fails the run, not the process.
    std::fs::write(&path, "system:\n  base: \"\"\n").unwrap();
    let err = built.run_once("third").await.unwrap_err();
    assert!(err.to_string().contains("system.base"), "{err}");
}

/// The explore child keeps its read-only policy, and summary calls from
/// children use the shared snapshot (regression guard for the assembly).
#[tokio::test]
async fn explore_children_stay_read_only() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("f.txt"), "needle here\n").unwrap();
    let model = ScriptedModel::new(vec![
        task_call("explore", "find the needle"),
        call("grep", r#"{"pattern":"needle"}"#),
        answer("found it"),
        answer("done"),
    ]);
    let built = agent(dir.path(), model.clone()).build().unwrap();
    let outcome = built.run_prompt("go", None).await.unwrap();
    assert_eq!(outcome.answer, "done");

    let requests = model.requests();
    assert!(!is_summary_request(&requests[0]));
    // The explore child's system prompt carries the vendored explore text.
    let child_system = requests[1].messages[0].content.clone().unwrap();
    assert!(
        child_system.contains("file search specialist"),
        "{child_system}"
    );
    let child_tools: Vec<&str> = requests[1]
        .tools
        .iter()
        .map(|spec| spec.function.name.as_str())
        .collect();
    assert_eq!(child_tools, ["read", "grep", "glob"], "{child_tools:?}");
    let task_result = outcome
        .messages
        .iter()
        .find(|message| message.role == mason::Role::Tool)
        .unwrap()
        .content
        .clone()
        .unwrap();
    assert!(task_result.contains("found it"), "{task_result}");
}

#[tokio::test]
async fn a_replaced_builtin_profile_widens_its_tools_and_keeps_its_text() {
    let dir = tempfile::tempdir().unwrap();
    let (site_fetch, calls) = EchoTool::boxed("site_fetch", "fetched");
    let model = ScriptedModel::new(vec![
        task_call("explore", "look at the live page"),
        call("site_fetch", "{}"),
        answer("child done"),
        answer("parent done"),
    ]);
    let built = agent(dir.path(), model.clone())
        .tool(site_fetch)
        .replace_profile(Profile {
            name: "explore".into(),
            description: String::new(),
            tools: Some(
                ["read", "grep", "glob", "skill", "site_fetch"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            ),
            ..Profile::subagent_profile(String::new(), String::new())
        })
        .build()
        .unwrap();
    let outcome = built.run_prompt("go", None).await.unwrap();
    assert_eq!(outcome.answer, "parent done");
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);

    let requests = model.requests();
    // The parent's task listing still carries the snapshot description for explore.
    let task_spec = requests[0]
        .tools
        .iter()
        .find(|spec| spec.function.name == "task")
        .expect("build may still delegate");
    let listing = task_spec.function.description.as_str();
    assert!(
        listing.contains("- explore: Fast agent specialized"),
        "{listing}"
    );

    // The child ran the widened toolset, with the vendored explore text.
    let child_system = requests[1].messages[0].content.clone().unwrap();
    assert!(
        child_system.contains("file search specialist"),
        "{child_system}"
    );
    let child_tools: Vec<&str> = requests[1]
        .tools
        .iter()
        .map(|spec| spec.function.name.as_str())
        .collect();
    assert!(child_tools.contains(&"site_fetch"), "{child_tools:?}");
    assert!(child_tools.contains(&"read"), "{child_tools:?}");
    assert!(!child_tools.contains(&"bash"), "{child_tools:?}");
}

#[tokio::test]
async fn replacing_an_unknown_profile_is_a_build_error() {
    let dir = tempfile::tempdir().unwrap();
    let model = ScriptedModel::new(vec![]);
    let error = agent(dir.path(), model)
        .replace_profile(Profile::new("nope", "not a builtin"))
        .build()
        .unwrap_err();
    assert!(
        error.to_string().contains("no profile named `nope`"),
        "{error}"
    );
}
