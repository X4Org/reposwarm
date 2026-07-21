# X4 RepoSwarm Fork Policy

This repository is a deliberately thin fork of
[`reposwarm/reposwarm`](https://github.com/reposwarm/reposwarm). X4 continues to
use the upstream CLI, API, UI, Temporal workflows, cache, prompt management, and
result hub. The forked worker exists only because upstream currently sends a
depth-three filename tree—not file contents—to the analysis model, which cannot
support truthful source and line citations.

## X4 delta

- Opt-in, bounded source-evidence bundles with deterministic file selection.
- Exact `relative/path.ext:line` prefixes for every supplied source line.
- Exclusion of dotenv, key/certificate, lock, generated, and binary files.
- Redaction of credential-like assignments and private-key blocks.
- Mandatory evidence instructions when a source bundle is present.
- Opt-in content-addressed prompt caching based on relevant section evidence,
  prompt/model policy, and dependency outputs instead of the whole commit.
- A non-secret `reposwarm:<repo>:<section>` request tag for usage accounting.
- X4-owned worker images at `ghcr.io/x4org/reposwarm-worker`.

All other RepoSwarm behavior should remain upstream-shaped. Do not add X4
scheduling, registry, GitHub publication, quality gating, or auto-merge policy
here; those responsibilities remain in `X4Org/x4`.

## Runtime controls

Source grounding is disabled by default. X4 enables and bounds it through:

```text
REPOSWARM_SOURCE_GROUNDING=true
REPOSWARM_SOURCE_BUNDLE_MAX_CHARS=120000
REPOSWARM_SOURCE_BUNDLE_MAX_FILES=120
REPOSWARM_SECTION_CACHE=true
```

Section caching is disabled by default. When enabled, broad overview, module,
entity, data-flow, and security prompts remain bound to all supplied source
evidence. Specialized prompts select a conservative evidence subset by path
and content keywords. Their actual upstream section context is part of the
identity, so a changed dependency invalidates downstream sections. Force mode
continues to bypass all prompt caches.

The X4 control plane must deploy an immutable worker digest and record it in the
reviewed RepoSwarm runtime lock. Never deploy this fork using an unverified
moving tag.

If GHCR visibility or host credentials temporarily prevent a control node from
pulling an already approved digest, manually dispatch CI with that digest. Its
short-lived amd64 OCI archive preserves the registry index digest when imported
through the control node's `moby` containerd namespace. This is a recovery path,
not a substitute for granting production hosts normal read access to the
package.

## Upstream reconciliation

1. Fetch `reposwarm/reposwarm` as the `upstream` remote.
2. Review upstream changes to repository analysis, prompt execution, worker
   configuration, and source handling before merging or rebasing.
3. Run the complete unit suite and the X4 source-grounding tests.
4. Build a commit-addressed worker image and canary it against Mastra.
5. Update the X4 runtime lock only after the canary passes.

If upstream gains equivalent bounded, secret-aware source evidence and citation
support, remove this delta and return X4 to the upstream worker image.
