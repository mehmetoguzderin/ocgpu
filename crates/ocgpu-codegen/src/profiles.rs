// SPDX-License-Identifier: CC0-1.0

use super::*;

pub(super) fn read_hip_runtime_profiles(
    workspace_root: &Path,
) -> Result<HipRuntimeProfiles, Error> {
    let path = workspace_root.join(HIP_RUNTIME_PROFILES_PATH);
    let source = read(&path)?;
    serde_json::from_str(&source).map_err(|source| Error::OracleJson { path, source })
}

pub(super) fn read_hip_runtime_declarations(
    workspace_root: &Path,
) -> Result<HipRuntimeDeclarations, Error> {
    let path = workspace_root.join(HIP_RUNTIME_DECLARATIONS_PATH);
    let source = read(&path)?;
    serde_json::from_str(&source).map_err(|source| Error::OracleJson { path, source })
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Validation(format!("{}: {}", HIP_RUNTIME_PROFILES_PATH, message.into()))
}

fn validate_hash(hash: &str, label: &str) -> Result<(), Error> {
    if hash.len() != 71
        || !hash.starts_with("sha256:")
        || !hash[7..].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(invalid(format!("{label} is not a canonical SHA-256")));
    }
    Ok(())
}

fn release_set<'a>(ledger: &'a HipRuntimeProfiles, name: &str) -> Result<&'a [String], Error> {
    ledger
        .release_sets
        .get(name)
        .map(Vec::as_slice)
        .ok_or_else(|| invalid(format!("unknown release set {name}")))
}

fn validate_platform(
    profile: &HipPlatformProfile,
    major: i32,
    names: &[&str],
    minimum: i32,
    proc_minimum: Option<i32>,
    raw_minimum: Option<i32>,
) -> Result<(), Error> {
    let maximum = (major + 1) * 10_000_000 - 1;
    if profile
        .library_candidates
        .iter()
        .map(String::as_str)
        .ne(names.iter().copied())
        || profile.runtime_version_min_inclusive != minimum
        || profile.runtime_version_max_inclusive != maximum
        || profile.proc_address_min_inclusive != proc_minimum
        || profile.raw_inventory_min_inclusive != raw_minimum
        || minimum / 10_000_000 != major
        || profile
            .proc_address_min_inclusive
            .is_some_and(|value| value < minimum || value > maximum)
        || profile
            .raw_inventory_min_inclusive
            .is_some_and(|value| value < minimum || value > maximum)
    {
        return Err(invalid(format!(
            "HIP {major} library/version/proc/raw interval is stale or unsafe"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
pub(super) fn validate_hip_runtime_profiles(
    manifest: &ApiManifest,
    ledger: &HipRuntimeProfiles,
    declarations: &HipRuntimeDeclarations,
) -> Result<(), Error> {
    if ledger.schema_version != 1
        || ledger.spdx_license_identifier != "CC0-1.0"
        || ledger.inventory_id != "hip-runtime-profiles"
        || ledger.scope.trim().is_empty()
        || ledger.runtime_version_encoding.expression != "major * 10000000 + minor * 100000 + patch"
        || !ledger
            .runtime_version_encoding
            .source
            .starts_with("https://")
        || ledger.compatibility_policy.rule.trim().is_empty()
        || !ledger.compatibility_policy.source.starts_with("https://")
        || ledger.compatibility_policy.fail_closed.trim().is_empty()
        || ledger.table_flag_encoding.zero_meaning.trim().is_empty()
    {
        return Err(invalid("metadata or fail-closed policy is stale"));
    }

    let manifest_flag = |name: &str| {
        manifest
            .constants
            .iter()
            .find(|constant| constant.name == name)
            .map(|constant| constant.value)
    };
    let flags = &ledger.table_flag_encoding;
    if (flags.mask, flags.shift, flags.hip5, flags.hip6, flags.hip7)
        != (0x00ff_0000, 16, 0x0005_0000, 0x0006_0000, 0x0007_0000)
        || manifest_flag("OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_MASK") != Some(i64::from(flags.mask))
        || manifest_flag("OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_5") != Some(i64::from(flags.hip5))
        || manifest_flag("OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_6") != Some(i64::from(flags.hip6))
        || manifest_flag("OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_7") != Some(i64::from(flags.hip7))
    {
        return Err(invalid(
            "profile flags disagree with the canonical API manifest",
        ));
    }
    if ledger.bootstrap_symbols != ["hipRuntimeGetVersion"] {
        return Err(invalid("bootstrap allowlist must be exact"));
    }

    let releases = [
        "hip-5.7.31541",
        "hip-5.7.31921",
        "hip-6.1.40093",
        "hip-6.2.41134",
        "hip-6.4.43484",
        "hip-7.2.53210",
        "hip-7.14.60850",
    ];
    for (name, expected) in [
        ("all_reviewed", releases.as_slice()),
        ("through_7_2", &releases[..6]),
        ("legacy_5_6", &releases[..5]),
        ("hip_7", &releases[5..]),
        ("only_7_14", &releases[6..]),
    ] {
        if release_set(ledger, name)?
            .iter()
            .map(String::as_str)
            .ne(expected.iter().copied())
        {
            return Err(invalid(format!("release set {name} is stale")));
        }
    }
    if ledger.release_sets.len() != 5 {
        return Err(invalid("unexpected release set"));
    }

    let expected_versions = [
        50_731_541, 50_731_921, 60_140_093, 60_241_134, 60_443_484, 70_253_210, 71_460_850,
    ];
    let expected_hashes = [
        [
            "sha256:d4677c1612f3e5eef5eca1815c8dd3571b8b00affaf0ee793994beaa2ab130ae",
            "sha256:c3081f64a2821a17e40ccb6fdd6dda19e38a69aacb6882ed9526e4ce26f1ab09",
            "sha256:d9e2297659156a4c6f7561cafbcea3e7e98893738228594827efef8474c1e23d",
            "sha256:56424963b77761088c00bcb7c229697b31ec3618354fafe9da5ec436c0b311ff",
            "sha256:2d440e43255a620278c496b4095816da1d0e8ad25a16745b5b9c3d4ab3894ead",
        ],
        [
            "sha256:d4677c1612f3e5eef5eca1815c8dd3571b8b00affaf0ee793994beaa2ab130ae",
            "sha256:c3081f64a2821a17e40ccb6fdd6dda19e38a69aacb6882ed9526e4ce26f1ab09",
            "sha256:d9e2297659156a4c6f7561cafbcea3e7e98893738228594827efef8474c1e23d",
            "sha256:a063ccfa37a6315d46850383466005363cad21fb96ac10e91e5d3de9186eb259",
            "sha256:2d440e43255a620278c496b4095816da1d0e8ad25a16745b5b9c3d4ab3894ead",
        ],
        [
            "sha256:7a4fcaebbe455eb96c020a6ab47a3400b36059838713b341ed2b927eb2a5ac74",
            "sha256:7f36ffc64b62b0255ca76a561be64a8481be6154330e31b23ede30bc0e226b23",
            "sha256:6d9cd2864bb4dc1b1b6a7e707c9b17ec601e5c8b4ae1a341ebecc948328227ae",
            "sha256:37056c8d1aa794139b84043fd5c4a44081ce4c945b317cb37cbb925b6d155869",
            "sha256:8bce3453eee0bc10cffe1dd1a326da33b852320704cbdc006ed2882dd2278aec",
        ],
        [
            "sha256:c4f7f295482dcea2578e18b7924e610c289c0ef8ae263c5f298b73f632ac3c1c",
            "sha256:1fdbcf68c43b541904b64c48e51c89f543970a62b36cf17bd29700b1835b3ff0",
            "sha256:652bcac90e1edebfe39b936524c82c080774fb4075a4d2acb6ced30b412b9590",
            "sha256:2a376e5605a42939a741bc28bfad950426adfbd094f592c54d9abcaa42c557cc",
            "sha256:547a53e1ee9b8c05723120c7894c7fee4eba2f77ec1576feeb3a3c4d7bbedea5",
        ],
        [
            "sha256:cfc8cb748d3bea5d0285819fda2d46f16718be901867ca52b9736c60b47e34ce",
            "sha256:747ea283542659391858971dbafc7ce16b82a017f4530aafd0d8b39515ce5bcc",
            "sha256:1e51532674aae5c0ee05bda30dc8907c7c547887c3cf00d8de202d36288c635b",
            "sha256:98cb8a9ac0357aefd7f1703c6027609995aa13a2bf92734434aaf462c02e2c28",
            "sha256:0b1646eceb5cfd517ee3aa937d853ea71dc5aab6cc7b128b7702c2d49ef3d1cb",
        ],
        [
            "sha256:af75c40c2777151e0dad3b7e960111d1793742ccb7d3eb5a0e1ffd2b181758f1",
            "sha256:16f713e8b5633ae434e59a3a92c0e521f435e5be7bd5952e7818fe9adf087eac",
            "sha256:af0a9a0af62e0b198df7ca4b570e7a158f7a73d2c23d964b27ef70426194c836",
            "sha256:9b50b1437d98a791fd03bf6a3c2bcbfb0f310fbb7487f6e4369182d206bc81e5",
            "sha256:338d7d54bba7882ccf3ef59b01668aa68fff3edf7847e5a241f230e758aa27ff",
        ],
        [
            "sha256:9d4221874dece55578c994ef8b06e8f607f5e033fb9ae701d198078f09f4ff94",
            "sha256:2074d65bbe51ee34773637b0c2a8c8afad4ab3ea2cb776535fbaf69aa107bafe",
            "sha256:c9a8cbee9a257d263dacc3c40cefe2d9519433b95fca8012cc38dbbad451bea5",
            "sha256:7bd8c8a717ae3d92905864b9806077d9873650006b94785ee2f2ea03802838a5",
            "sha256:56bd3b1ca64010038dd493e711ad77d166bd09b9f47e99431129712da60af392",
        ],
    ];
    if ledger.reviewed_releases.len() != releases.len() {
        return Err(invalid("reviewed release count is stale"));
    }
    for (((release, id), version), hashes) in ledger
        .reviewed_releases
        .iter()
        .zip(releases)
        .zip(expected_versions)
        .zip(expected_hashes)
    {
        let actual_hashes = [
            release.hip_archive_sha256.as_str(),
            release.hip_header_sha256.as_str(),
            release.hip_version_sha256.as_str(),
            release.clr_archive_sha256.as_str(),
            release.clr_cmake_sha256.as_str(),
        ];
        if release.id != id
            || release.runtime_version != version
            || actual_hashes != hashes
            || release.rocm_release.trim().is_empty()
            || release.hip_commit.trim().is_empty()
            || release.clr_commit.trim().is_empty()
            || !release
                .hip_archive_url
                .starts_with("https://github.com/ROCm/HIP/archive/")
            || !release
                .clr_archive_url
                .starts_with("https://github.com/ROCm/clr/archive/")
            || release.hip_header_path != "include/hip/hip_runtime_api.h"
            || release.clr_cmake_path != "hipamd/src/CMakeLists.txt"
        {
            return Err(invalid(format!("source facts for {id} are stale")));
        }
        for hash in actual_hashes {
            validate_hash(hash, id)?;
        }
        if id == "hip-5.7.31541" {
            let observed = release
                .observed_runtime
                .as_ref()
                .ok_or_else(|| invalid("HIP 5.7.31541 lacks observed-runtime evidence"))?;
            validate_hash(&observed.sha256, "observed HIP runtime")?;
            if observed.platform != "x86_64-pc-windows-msvc"
                || observed.library != "C:/Windows/System32/amdhip64.dll"
                || observed.sha256
                    != "sha256:f6a64adfef5336490b530942cd2b22fa5d7c07d20c0d4cfb01e5b8c93ba20c94"
                || observed.hip_runtime_get_version != version
                || observed.file_version != "10.0.3584.0"
                || observed.product_version != "10.0.3584.0"
                || observed.signature_status != "Valid"
                || !observed
                    .signer_subject
                    .starts_with("CN=Microsoft Windows Hardware Compatibility Publisher")
                || observed.signer_thumbprint != "32B4E7CE92A59C722B8BDF7AE3E66520CFB9DE21"
                || !observed.scope_note.contains("not a claim")
            {
                return Err(invalid("HIP 5.7.31541 runtime observation is stale"));
            }
        } else if release.observed_runtime.is_some() {
            return Err(invalid(format!(
                "{id} has an unexpected machine-local runtime observation"
            )));
        }
        let expected_proc = version >= 60_241_134;
        if release.proc_address_declared != expected_proc {
            return Err(invalid(format!("hipGetProcAddress fact for {id} is stale")));
        }
    }

    let profile_facts = [
        (7, 0x0007_0000, &releases[5..], 70_253_210, Some(70_253_210)),
        (
            6,
            0x0006_0000,
            &releases[2..5],
            60_140_093,
            Some(60_241_134),
        ),
        (5, 0x0005_0000, &releases[..2], 50_731_541, None),
    ];
    if ledger.profiles.len() != profile_facts.len() {
        return Err(invalid("expected exactly HIP 7/6/5 profiles"));
    }
    for (profile, (major, flag, reviewed, minimum, proc_minimum)) in
        ledger.profiles.iter().zip(profile_facts)
    {
        let adapters = if major == 7 {
            &[][..]
        } else {
            &["hipMemcpyHtoD"][..]
        };
        if profile.runtime_major != major
            || profile.table_flag != flag
            || profile
                .reviewed_release_ids
                .iter()
                .map(String::as_str)
                .ne(reviewed.iter().copied())
            || profile
                .common_adapter_symbols
                .iter()
                .map(String::as_str)
                .ne(adapters.iter().copied())
        {
            return Err(invalid(format!("HIP {major} descriptor is stale")));
        }
        let windows_names = match major {
            5 => &["amdhip64.dll"][..],
            6 => &["amdhip64_6.dll"][..],
            7 => &["amdhip64_7.dll"][..],
            _ => unreachable!(),
        };
        let linux_names = match major {
            5 => &["libamdhip64.so.5", "libamdhip64.so"][..],
            6 => &["libamdhip64.so.6", "libamdhip64.so"][..],
            7 => &["libamdhip64.so.7", "libamdhip64.so"][..],
            _ => unreachable!(),
        };
        validate_platform(
            &profile.windows,
            major,
            windows_names,
            minimum,
            proc_minimum,
            (major == 7).then_some(70_253_210),
        )?;
        validate_platform(
            &profile.linux,
            major,
            linux_names,
            minimum,
            proc_minimum,
            (major == 7).then_some(71_460_850),
        )?;
    }

    if declarations.schema_version != 1
        || declarations.spdx_license_identifier != "CC0-1.0"
        || declarations.inventory_id != "hip-runtime-profile-declarations"
        || declarations.provenance.trim().is_empty()
        || declarations.snapshots.len() != releases.len()
    {
        return Err(invalid(
            "committed per-release declaration evidence is stale",
        ));
    }
    let expected_type_facts = [
        (
            "hipError_t",
            "type",
            "type hipError_t=enum hipError_t",
            "sha256:5a0e0c422fcb0ff5f6631cffd341f3f7687f276007ca2e2a8b70ffc0afeb9e16",
        ),
        (
            "hipDevice_t",
            "type",
            "type hipDevice_t=int",
            "sha256:9a20994e8b9f2756d5df7b5fea7bb4f5de7cb6f7c0892482f7f414a84c4a93a1",
        ),
        (
            "hipDeviceAttribute_t",
            "type",
            "type hipDeviceAttribute_t=enum hipDeviceAttribute_t",
            "sha256:f6d1175e422ab473ec68cfd0b2a43ac73a6e264f64f149caee93353cc6c53e4f",
        ),
        (
            "hipDeviceptr_t",
            "opaque_handle",
            "type hipDeviceptr_t=void*",
            "sha256:0ab92e8fd458f15efe96283195f879f64a41bdb81ebee9deaa3044e91cd5a655",
        ),
        (
            "hipCtx_t",
            "opaque_handle",
            "type hipCtx_t=struct ihipCtx_t*",
            "sha256:163c1033e30e3fd42da45f3b40f56bcf5156703dbdac1824c565a83ecca26dcb",
        ),
        (
            "hipStream_t",
            "opaque_handle",
            "type hipStream_t=struct ihipStream_t*",
            "sha256:ed72cbf75ff590a55f1f48a1c46df404e7b76844c9475108f721d58c6f9bbe1f",
        ),
        (
            "hipEvent_t",
            "opaque_handle",
            "type hipEvent_t=struct ihipEvent_t*",
            "sha256:e2cc167a459c5c724e35e864685ec6b344991a77e1cc3b2580025d2b153a2adc",
        ),
        (
            "hipModule_t",
            "opaque_handle",
            "type hipModule_t=struct ihipModule_t*",
            "sha256:1c55fbfab9fd1817e7bd2fea1b99a27337cc8438addbd4a9f00f936e4d2dc1e3",
        ),
        (
            "hipFunction_t",
            "opaque_handle",
            "type hipFunction_t=struct ihipModuleSymbol_t*",
            "sha256:0aea881fac4d9f4884ff6c43f546c94900441c5ef78141d27f68e97e9d1b36e0",
        ),
    ];
    let mut declaration_functions = BTreeMap::new();
    for ((snapshot, release), ledger_release) in declarations
        .snapshots
        .iter()
        .zip(releases)
        .zip(&ledger.reviewed_releases)
    {
        let (expected_inventory_id, expected_header_role, expected_platforms): (
            &str,
            &str,
            &[&str],
        ) = match release {
            "hip-5.7.31541" | "hip-5.7.31921" => (
                "hip-profile-5.7.1-review",
                "authoritative-hip-header",
                &[
                    "aarch64-unknown-linux-gnu",
                    "x86_64-pc-windows-msvc",
                    "x86_64-unknown-linux-gnu",
                ],
            ),
            "hip-6.1.40093" => (
                "hip-profile-6.1.2-review",
                "authoritative-hip-header",
                &[
                    "aarch64-unknown-linux-gnu",
                    "x86_64-pc-windows-msvc",
                    "x86_64-unknown-linux-gnu",
                ],
            ),
            "hip-6.2.41134" => (
                "hip-profile-6.2.4-review",
                "authoritative-hip-header",
                &[
                    "aarch64-unknown-linux-gnu",
                    "x86_64-pc-windows-msvc",
                    "x86_64-unknown-linux-gnu",
                ],
            ),
            "hip-6.4.43484" => (
                "hip-profile-6.4.2-review",
                "authoritative-hip-header",
                &[
                    "aarch64-unknown-linux-gnu",
                    "x86_64-pc-windows-msvc",
                    "x86_64-unknown-linux-gnu",
                ],
            ),
            "hip-7.2.53210" => (
                "hip-profile-7.2.53210-review",
                "authoritative-hip-header",
                &[
                    "aarch64-unknown-linux-gnu",
                    "x86_64-pc-windows-msvc",
                    "x86_64-unknown-linux-gnu",
                ],
            ),
            "hip-7.14.60850" => (
                "hip-profile-7.14.60850-review",
                "semantic-hip-header",
                &["aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu"],
            ),
            _ => unreachable!(),
        };
        if snapshot.release_id != release
            || snapshot.source_inventory_id != expected_inventory_id
            || snapshot
                .source_inventory_platforms
                .iter()
                .map(String::as_str)
                .ne(expected_platforms.iter().copied())
            || snapshot.source_header_artifact.role != expected_header_role
            || snapshot.source_header_artifact.url != ledger_release.hip_archive_url
            || snapshot.source_header_artifact.sha256 != ledger_release.hip_header_sha256
            || snapshot.source_header_artifact.path != ledger_release.hip_header_path
            || snapshot.source_header_artifact.revision.trim().is_empty()
            || snapshot.target_abi.pointer_width_bits != 64
            || snapshot.target_abi.size_t_width_bits != 64
            || snapshot.target_abi.enum_width_bits != 32
            || snapshot.target_abi.success_value != 0
            || snapshot.target_abi.null_pointer_sentinel != "all-bits-zero"
            || snapshot.functions.len() != 27
            || snapshot.transitive_types.len() != expected_type_facts.len()
            || snapshot.device_attributes.len() != 32
        {
            return Err(invalid(format!(
                "{release} compact declaration snapshot is incomplete"
            )));
        }
        let function_map = snapshot
            .functions
            .iter()
            .map(|entry| (entry.name.as_str(), entry))
            .collect::<BTreeMap<_, _>>();
        if function_map.len() != 27
            || function_map.values().any(|entry| {
                entry.normalized_signature.trim().is_empty()
                    || entry
                        .platforms
                        .iter()
                        .map(String::as_str)
                        .ne(expected_platforms.iter().copied())
            })
        {
            return Err(invalid(format!(
                "{release} function declaration/platform evidence is stale"
            )));
        }
        let bootstrap = function_map
            .get("hipRuntimeGetVersion")
            .ok_or_else(|| invalid(format!("{release} lacks the bootstrap declaration")))?;
        if bootstrap.normalized_signature
            != "fn hipRuntimeGetVersion[abi=C](runtimeversion:int*:mut:out:nullable=unknown)->hipError_t"
            || bootstrap.signature_hash
                != "sha256:6368151ecb11fe275166ad51aec2aa82ff0e50607cdd47f260bca92f2c1ae90f"
        {
            return Err(invalid(format!("{release} bootstrap declaration is stale")));
        }
        for (actual, expected) in snapshot.transitive_types.iter().zip(expected_type_facts) {
            if (
                actual.name.as_str(),
                actual.kind.as_str(),
                actual.normalized_signature.as_str(),
                actual.signature_hash.as_str(),
            ) != expected
                || actual
                    .platforms
                    .iter()
                    .map(String::as_str)
                    .ne(expected_platforms.iter().copied())
            {
                return Err(invalid(format!(
                    "{release} transitive type {} is stale",
                    actual.name
                )));
            }
        }
        for (actual, expected) in snapshot
            .device_attributes
            .iter()
            .zip(&ledger.device_attributes)
        {
            validate_hash(&actual.signature_hash, &actual.name)?;
            if actual.name != expected.name
                || actual.value != expected.value
                || !actual
                    .normalized_signature
                    .starts_with(&format!("enum-value {}=", actual.name))
                || actual
                    .platforms
                    .iter()
                    .map(String::as_str)
                    .ne(expected_platforms.iter().copied())
            {
                return Err(invalid(format!(
                    "{release} device-attribute {} evidence is stale",
                    expected.name
                )));
            }
        }
        declaration_functions.insert(release, function_map);
    }
    let all_release_ids = releases.into_iter().collect::<BTreeSet<_>>();
    let mut exact_names = BTreeSet::new();
    for function in &ledger.common_functions {
        if !exact_names.insert(function.name.as_str()) {
            return Err(invalid(format!("duplicate function {}", function.name)));
        }
        let mut covered = BTreeSet::new();
        let groups = std::iter::once((
            function.release_set.as_str(),
            function.signature_hash.as_str(),
        ))
        .chain(
            function
                .additional_signatures
                .iter()
                .map(|item| (item.release_set.as_str(), item.signature_hash.as_str())),
        );
        for (set, hash) in groups {
            validate_hash(hash, &function.name)?;
            for release in release_set(ledger, set)? {
                if !covered.insert(release.as_str()) {
                    return Err(invalid(format!(
                        "{} has overlapping release signatures",
                        function.name
                    )));
                }
                let expected = declaration_functions
                    .get(release.as_str())
                    .and_then(|functions| functions.get(function.name.as_str()))
                    .ok_or_else(|| {
                        invalid(format!(
                            "{release} compact snapshot lacks {}",
                            function.name
                        ))
                    })?;
                if expected.signature_hash != hash {
                    return Err(invalid(format!(
                        "{} signature is stale for {release}",
                        function.name
                    )));
                }
            }
        }
        if covered != all_release_ids {
            return Err(invalid(format!(
                "{} does not cover all reviewed releases",
                function.name
            )));
        }
    }
    if ledger.common_adapters.len() != 1 {
        return Err(invalid("expected one common adapter"));
    }
    let adapter = &ledger.common_adapters[0];
    let adapter_facts = [
        (
            "legacy_5_6",
            "fn hipMemcpyHtoD[abi=C](dst:hipdeviceptr_t:mut:in:nullable=unknown,src:void*:mut:in:nullable=unknown,sizebytes:size_t:value:in:nullable=false)->hipError_t",
            "sha256:c8c4b30955f248b45d00bc89899e7d853525cb902c70b3e79859f2696268cfb9",
        ),
        (
            "hip_7",
            "fn hipMemcpyHtoD[abi=C](dst:hipdeviceptr_t:mut:in:nullable=unknown,src:const void*:const:in:nullable=unknown,sizebytes:size_t:value:in:nullable=false)->hipError_t",
            "sha256:723df21271bdca7810cef614dcde20639ed6610392f169a165f00e15f7dec103",
        ),
    ];
    if adapter.name != "hipMemcpyHtoD"
        || adapter.adapter.trim().is_empty()
        || adapter.signature_variants.len() != adapter_facts.len()
    {
        return Err(invalid("HtoD adapter evidence is incomplete"));
    }
    let mut adapter_covered = BTreeSet::new();
    for (actual, expected) in adapter.signature_variants.iter().zip(adapter_facts) {
        if (
            actual.release_set.as_str(),
            actual.normalized_signature.as_str(),
            actual.signature_hash.as_str(),
        ) != expected
        {
            return Err(invalid("HtoD adapter signature evidence is stale"));
        }
        for release in release_set(ledger, &actual.release_set)? {
            let declaration = declaration_functions
                .get(release.as_str())
                .and_then(|functions| functions.get(adapter.name.as_str()))
                .ok_or_else(|| {
                    invalid(format!("{release} compact snapshot lacks {}", adapter.name))
                })?;
            if declaration.normalized_signature != actual.normalized_signature
                || declaration.signature_hash != actual.signature_hash
            {
                return Err(invalid(format!(
                    "{release} HtoD adapter declaration is stale"
                )));
            }
            adapter_covered.insert(release.as_str());
        }
    }
    if adapter_covered != all_release_ids {
        return Err(invalid("HtoD adapter release coverage is incomplete"));
    }

    let manifest_common = manifest
        .functions
        .iter()
        .map(|function| function.hip.vendor_symbol.as_str())
        .collect::<BTreeSet<_>>();
    exact_names.insert(adapter.name.as_str());
    if exact_names != manifest_common || exact_names.len() != 26 {
        return Err(invalid(
            "exact+adapter allowlist disagrees with the common ABI",
        ));
    }
    let mut attributes = BTreeSet::new();
    for attribute in &ledger.device_attributes {
        if !attributes.insert(attribute.name.as_str()) {
            return Err(invalid(format!("duplicate attribute {}", attribute.name)));
        }
        let constant = manifest.constants.iter().find(|constant| {
            constant.backend.as_deref() == Some("hip")
                && constant.vendor_name.as_deref() == Some(attribute.name.as_str())
        });
        if constant.map(|constant| constant.value) != Some(attribute.value) {
            return Err(invalid(format!(
                "attribute {} value is stale",
                attribute.name
            )));
        }
    }
    if attributes.len() != 32
        || ledger.transitive_abi_facts.len() < 8
        || ledger.transitive_abi_facts.iter().any(|fact| {
            fact.fact.trim().is_empty() || fact.proof.trim().is_empty() || fact.expected.is_null()
        })
        || ledger.semantic_reviews.len() < 5
        || ledger.semantic_reviews.iter().any(|review| {
            review.operations.is_empty()
                || review.finding.trim().is_empty()
                || review.proof.trim().is_empty()
        })
        || ledger.library_naming_evidence.len() != 6
        || ledger.library_naming_evidence.iter().any(|evidence| {
            !matches!(evidence.platform.as_str(), "windows" | "linux")
                || !(5..=7).contains(&evidence.major)
                || evidence.names.is_empty()
                || !evidence.source.starts_with("https://")
        })
    {
        return Err(invalid(
            "attribute, transitive ABI, semantic, or library evidence is incomplete",
        ));
    }
    Ok(())
}

fn rust_slice(values: &[&str]) -> String {
    let mut output = String::from("&[");
    for value in values {
        write!(output, "\n    {value:?},").expect("String write");
    }
    output.push_str("\n]");
    output
}

fn version_literal(value: i32) -> String {
    format!(
        "{}_{:03}_{:03}",
        value / 1_000_000,
        (value / 1_000) % 1_000,
        value % 1_000
    )
}

fn optional_version_literal(value: Option<i32>) -> String {
    value.map_or_else(
        || "None".to_owned(),
        |value| format!("Some({})", version_literal(value)),
    )
}

fn platform_source(profile: &HipPlatformProfile) -> String {
    let names = profile
        .library_candidates
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    format!(
        "HipPlatformRuntimeProfile {{\n    library_candidates: {},\n    runtime_version_min_inclusive: {},\n    runtime_version_max_inclusive: {},\n    proc_address_min_inclusive: {},\n    raw_inventory_min_inclusive: {},\n}}",
        rust_slice(&names),
        version_literal(profile.runtime_version_min_inclusive),
        version_literal(profile.runtime_version_max_inclusive),
        optional_version_literal(profile.proc_address_min_inclusive),
        optional_version_literal(profile.raw_inventory_min_inclusive),
    )
}

pub(super) fn render_hip_runtime_profiles(
    manifest: &ApiManifest,
    ledger: &HipRuntimeProfiles,
) -> String {
    let common = manifest
        .functions
        .iter()
        .map(|function| function.hip.vendor_symbol.as_str())
        .collect::<Vec<_>>();
    let legacy_exact = common
        .iter()
        .copied()
        .filter(|name| *name != "hipMemcpyHtoD")
        .collect::<Vec<_>>();
    let bootstrap = ledger
        .bootstrap_symbols
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let mut output = format!(
        "// SPDX-License-Identifier: CC0-1.0\n\n//! Generated from `oracle/vendor/hip/runtime-profiles.json`; do not edit.\n\n\
         /// One platform-specific interval in a reviewed HIP runtime ABI profile.\n#[derive(Clone, Copy, Debug, Eq, PartialEq)]\n\
         pub(crate) struct HipPlatformRuntimeProfile {{\n    /// Library basenames tried in fail-closed preference order.\n    pub library_candidates: &'static [&'static str],\n    /// Smallest accepted `hipRuntimeGetVersion` value.\n    pub runtime_version_min_inclusive: i32,\n    /// Largest accepted `hipRuntimeGetVersion` value.\n    pub runtime_version_max_inclusive: i32,\n    /// Smallest reviewed version with `hipGetProcAddress`, if any.\n    pub proc_address_min_inclusive: Option<i32>,\n    /// Smallest version allowed to populate the exhaustive raw inventory.\n    pub raw_inventory_min_inclusive: Option<i32>,\n}}\n\n\
         /// Reviewed compatibility facts for one HIP runtime ABI major.\n#[derive(Clone, Copy, Debug, Eq, PartialEq)]\n\
         pub(crate) struct HipRuntimeProfileDescriptor {{\n    /// Major decoded from `hipRuntimeGetVersion`.\n    pub runtime_major: i32,\n    /// Bits advertised through common and raw table flags.\n    pub table_flag: u32,\n    /// Symbols available through the common profile.\n    pub common_symbols: &'static [&'static str],\n    /// Common symbols with identical normalized declarations.\n    pub raw_exact_symbols: &'static [&'static str],\n    /// Common symbols reached through a reviewed adapter.\n    pub common_adapter_symbols: &'static [&'static str],\n    /// Symbols callable before profile selection.\n    pub bootstrap_symbols: &'static [&'static str],\n    /// Windows compatibility facts.\n    pub windows: HipPlatformRuntimeProfile,\n    /// Linux compatibility facts.\n    pub linux: HipPlatformRuntimeProfile,\n}}\n\n\
         /// Bootstrap calls made before profile selection.\npub(crate) const HIP_BOOTSTRAP_SYMBOLS: &[&str] = {};\n\n\
         /// Common subset available in every profile.\npub(crate) const HIP_COMMON_PROFILE_SYMBOLS: &[&str] = {};\n\n\
         /// Legacy common subset with identical declarations.\npub(crate) const HIP_LEGACY_COMMON_RAW_EXACT_SYMBOLS: &[&str] = {};\n\n\
         /// Legacy calls requiring an explicit adapter.\npub(crate) const HIP_COMMON_ADAPTER_SYMBOLS: &[&str] = &[\"hipMemcpyHtoD\"];\n\n\
         /// HIP 7 uses current declarations for the complete common subset.\npub(crate) const HIP7_COMMON_RAW_EXACT_SYMBOLS: &[&str] = HIP_COMMON_PROFILE_SYMBOLS;\n\n\
         /// HIP 7 needs no common-call adapter.\npub(crate) const HIP7_COMMON_ADAPTER_SYMBOLS: &[&str] = &[];\n\n\
         /// Closed set of supported profiles, newest first.\npub(crate) const HIP_RUNTIME_PROFILES: &[HipRuntimeProfileDescriptor] = &[\n",
        rust_slice(&bootstrap),
        rust_slice(&common),
        rust_slice(&legacy_exact),
    );
    for profile in &ledger.profiles {
        let (exact, adapters) = if profile.common_adapter_symbols.is_empty() {
            (
                "HIP7_COMMON_RAW_EXACT_SYMBOLS",
                "HIP7_COMMON_ADAPTER_SYMBOLS",
            )
        } else {
            (
                "HIP_LEGACY_COMMON_RAW_EXACT_SYMBOLS",
                "HIP_COMMON_ADAPTER_SYMBOLS",
            )
        };
        writeln!(
            output,
            "    HipRuntimeProfileDescriptor {{\n        runtime_major: {},\n        table_flag: ocgpu_abi::OCGPU_API_FLAG_HIP_RUNTIME_PROFILE_{},\n        common_symbols: HIP_COMMON_PROFILE_SYMBOLS,\n        raw_exact_symbols: {exact},\n        common_adapter_symbols: {adapters},\n        bootstrap_symbols: HIP_BOOTSTRAP_SYMBOLS,\n        windows: {},\n        linux: {},\n    }},",
            profile.runtime_major,
            profile.runtime_major,
            platform_source(&profile.windows),
            platform_source(&profile.linux),
        )
        .expect("String write");
    }
    output.push_str("];\n");
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs() -> (ApiManifest, HipRuntimeProfiles, HipRuntimeDeclarations) {
        (
            toml::from_str(include_str!("../../../api/ocgpu-api.toml"))
                .expect("canonical API manifest parses"),
            serde_json::from_str(include_str!(
                "../../../oracle/vendor/hip/runtime-profiles.json"
            ))
            .expect("HIP profile ledger parses"),
            serde_json::from_str(include_str!(
                "../../../oracle/vendor/hip/runtime-profile-declarations.json"
            ))
            .expect("HIP profile declarations parse"),
        )
    }

    #[test]
    fn reviewed_profile_declarations_validate() {
        let (manifest, ledger, declarations) = inputs();
        validate_hip_runtime_profiles(&manifest, &ledger, &declarations)
            .expect("reviewed profile evidence is coherent");
    }

    #[test]
    fn profile_source_binding_and_platform_drift_fail_closed() {
        let (manifest, ledger, mut declarations) = inputs();
        declarations.snapshots[5].source_header_artifact.sha256 =
            format!("sha256:{}", "0".repeat(64));
        assert!(validate_hip_runtime_profiles(&manifest, &ledger, &declarations).is_err());

        let (manifest, ledger, mut declarations) = inputs();
        declarations.snapshots[5]
            .source_inventory_platforms
            .remove(0);
        assert!(validate_hip_runtime_profiles(&manifest, &ledger, &declarations).is_err());
    }

    #[test]
    fn profile_device_attribute_drift_fails_closed() {
        let (manifest, ledger, mut declarations) = inputs();
        declarations.snapshots[0].device_attributes[0].value += 1;
        assert!(validate_hip_runtime_profiles(&manifest, &ledger, &declarations).is_err());
    }
}
