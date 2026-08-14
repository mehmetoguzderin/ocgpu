// SPDX-License-Identifier: CC0-1.0

//! Public HIP runtime-profile contracts that require no GPU runtime.

use ocgpu::{
    ApiMetadata, BackendDiagnostics, BackendKind, HipRuntimeProfile, RuntimeSymbolStatus,
    SymbolResolution,
};

#[test]
fn profile_flags_round_trip_through_public_metadata() {
    for profile in [
        HipRuntimeProfile::Hip5,
        HipRuntimeProfile::Hip6,
        HipRuntimeProfile::Hip7,
    ] {
        let metadata = ApiMetadata {
            backend: BackendKind::Hip,
            abi_version: ocgpu::sys::OCGPU_ABI_VERSION_1,
            flags: profile.api_flags(),
            driver_version: 0,
            struct_size: 0,
        };
        assert_eq!(metadata.hip_runtime_profile(), Some(profile));
    }

    let cuda_metadata = ApiMetadata {
        backend: BackendKind::Cuda,
        abi_version: ocgpu::sys::OCGPU_ABI_VERSION_1,
        flags: HipRuntimeProfile::Hip7.api_flags(),
        driver_version: 0,
        struct_size: 0,
    };
    assert_eq!(cuda_metadata.hip_runtime_profile(), None);
}

#[test]
fn profile_omissions_are_distinct_from_missing_optional_exports() {
    let status = |name, resolution| RuntimeSymbolStatus {
        name,
        resolved_name: None,
        resolution,
        proc_attempts: 0,
        required: false,
        applicable: true,
    };
    let diagnostics = BackendDiagnostics {
        backend: BackendKind::Hip,
        library_path: "mock-amdhip64_6".into(),
        runtime_version: Some(60_400_000),
        driver_version: Some(60_400_000),
        compiled_api_version: 70_200_000,
        hip_runtime_profile: Some(HipRuntimeProfile::Hip6),
        proc_address_support: false,
        proc_address_variant: None,
        loaded_architecture: "x86_64",
        symbols: vec![
            status("hipReviewedCore", SymbolResolution::Direct),
            status("hipAbsentExport", SymbolResolution::Missing),
            status("hipNewerProfileOnly", SymbolResolution::ProfileUnavailable),
            RuntimeSymbolStatus {
                name: "hipMemcpyHtoD",
                resolved_name: Some("hipMemcpyHtoD"),
                resolution: SymbolResolution::DirectAdapter,
                proc_attempts: 0,
                required: true,
                applicable: true,
            },
        ],
    };

    assert_eq!(diagnostics.missing_optional_symbols().count(), 1);
    assert_eq!(diagnostics.missing_required_symbols().count(), 0);
    assert_eq!(diagnostics.profile_omissions().count(), 2);
    assert_eq!(diagnostics.missing_symbols().count(), 3);
}
