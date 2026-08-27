# SPDX-License-Identifier: CC0-1.0

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$violations = [Collections.Generic.List[string]]::new()

$sdkRootVariables = @(
    'CUDA_HOME',
    'CUDA_PATH',
    'CUDA_ROOT',
    'CUDAToolkit_ROOT',
    'HIP_HOME',
    'HIP_PATH',
    'HIP_ROOT_DIR',
    'ROCM_HOME',
    'ROCM_PATH',
    'ROCM_ROOT'
)
foreach ($name in $sdkRootVariables) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if (-not [string]::IsNullOrWhiteSpace($value)) {
        $violations.Add("SDK environment root $name=$value")
    }
}

$sdkCommands = @(
    'compute-sanitizer',
    'cuobjdump',
    'cuda-gdb',
    'fatbinary',
    'hip-config',
    'hipcc',
    'hipconfig',
    'nvcc',
    'nvprune',
    'ptxas'
)
foreach ($name in $sdkCommands) {
    $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $command) {
        $location = if ([string]::IsNullOrWhiteSpace($command.Source)) {
            $command.Definition
        } else {
            $command.Source
        }
        $violations.Add("SDK compiler/configuration tool $name at $location")
    }
}

$runningOnWindows = [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT
$sdkArtifactPatterns = if ($runningOnWindows) {
    @(
        "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\*\include\cuda.h",
        "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\*\include\nvrtc.h",
        "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\*\lib\x64\cuda.lib",
        "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\*\lib\x64\cudart.lib",
        "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\*\lib\x64\*nvrtc*.lib",
        "$env:ProgramFiles\AMD\ROCm\*\include\hip\hip_runtime_api.h",
        "$env:ProgramFiles\AMD\ROCm\*\include\hip\hiprtc.h",
        "$env:ProgramFiles\AMD\ROCm\*\lib\amdhip64.lib",
        "$env:ProgramFiles\AMD\ROCm\*\lib\*hiprtc*.lib"
    )
} else {
    @(
        '/opt/cuda/include/cuda.h',
        '/opt/cuda/include/nvrtc.h',
        '/opt/cuda/lib64/libnvrtc*.a',
        '/opt/cuda/targets/*/include/cuda.h',
        '/opt/cuda/targets/*/include/nvrtc.h',
        '/opt/cuda/targets/*/lib/libnvrtc*.a',
        '/opt/rocm/include/hip/hip_runtime_api.h',
        '/opt/rocm/include/hip/hiprtc.h',
        '/opt/rocm/lib/cmake/hip/hip-config.cmake',
        '/opt/rocm/lib/libhiprtc*.a',
        '/opt/rocm/lib64/libhiprtc*.a',
        '/opt/rocm-*/include/hip/hip_runtime_api.h',
        '/opt/rocm-*/include/hip/hiprtc.h',
        '/opt/rocm-*/lib/cmake/hip/hip-config.cmake',
        '/opt/rocm-*/lib/libhiprtc*.a',
        '/opt/rocm-*/lib64/libhiprtc*.a',
        '/usr/include/cuda.h',
        '/usr/include/nvrtc.h',
        '/usr/include/hip/hip_runtime_api.h',
        '/usr/include/hip/hiprtc.h',
        '/usr/lib*/libnvrtc*.a',
        '/usr/lib*/libhiprtc*.a',
        '/usr/lib/*/libnvrtc*.a',
        '/usr/lib/*/libhiprtc*.a',
        '/usr/local/cuda/include/cuda.h',
        '/usr/local/cuda/include/nvrtc.h',
        '/usr/local/cuda/lib64/libnvrtc*.a',
        '/usr/local/cuda/lib64/stubs/libcuda.so',
        '/usr/local/cuda/targets/*/include/cuda.h',
        '/usr/local/cuda/targets/*/include/nvrtc.h',
        '/usr/local/cuda/targets/*/lib/libnvrtc*.a',
        '/usr/local/include/nvrtc.h',
        '/usr/local/include/hip/hip_runtime_api.h',
        '/usr/local/include/hip/hiprtc.h',
        '/usr/local/lib*/libnvrtc*.a',
        '/usr/local/lib*/libhiprtc*.a',
        '/usr/local/lib/*/libnvrtc*.a',
        '/usr/local/lib/*/libhiprtc*.a'
    )
}
foreach ($pattern in $sdkArtifactPatterns) {
    foreach ($match in @(Get-Item -Path $pattern -ErrorAction SilentlyContinue)) {
        $violations.Add("SDK header/link-time artifact $($match.FullName)")
    }
}

$trackedFiles = @(& git ls-files)
if ($LASTEXITCODE -ne 0) {
    throw "git ls-files failed with exit code $LASTEXITCODE"
}
$vendorBinaryName = '^(?:lib)?(?:cuda|cudart|nvrtc(?:-builtins)?|nvcuda|amdhip64|hiprtc|opencl)[A-Za-z0-9_.-]*[.](?:a|lib|dll|so(?:[.][0-9.]+)?)$'
foreach ($relativePath in $trackedFiles) {
    $fileName = [IO.Path]::GetFileName($relativePath)
    if ($fileName -match $vendorBinaryName) {
        $violations.Add("tracked vendor binary $relativePath")
    }
}

$linkInputExtensions = @(
    '.bat', '.c', '.cc', '.cmake', '.cpp', '.h', '.ps1', '.rs', '.sh',
    '.toml', '.yaml', '.yml'
)
$vendorLinkDirective = '(?im)(?:#\s*\[\s*link\s*\([^\]]*name\s*=\s*"(?:nvrtc|hiprtc)[A-Za-z0-9_.-]*"[^\]]*\]|rustc-link-(?:lib|arg)[^\r\n]*(?:nvrtc|hiprtc)|"-l"\s*,\s*"(?:static=|dylib=)?(?:nvrtc|hiprtc)[A-Za-z0-9_.-]*"|(?:^|[\s"])-l(?:nvrtc|hiprtc)[A-Za-z0-9_.-]*(?:[\s"]|$)|/DEFAULTLIB:(?:nvrtc|hiprtc)[A-Za-z0-9_.-]*(?:[.]lib)?|(?:^|[\s"''/\\])(?:lib)?(?:nvrtc|hiprtc)[A-Za-z0-9_.-]*[.](?:a|lib)(?:[\s"''/\\]|$)|target_link_libraries\s*\([^\r\n]*(?:nvrtc|hiprtc))'
foreach ($relativePath in $trackedFiles) {
    if ($relativePath -eq '.github/scripts/assert-sdk-free.ps1') {
        continue
    }
    $fileName = [IO.Path]::GetFileName($relativePath)
    $extension = [IO.Path]::GetExtension($relativePath)
    if (($extension -notin $linkInputExtensions) -and
        ($fileName -notin @('CMakeLists.txt', 'Makefile'))) {
        continue
    }
    $content = Get-Content -LiteralPath $relativePath -Raw
    if ($content -match $vendorLinkDirective) {
        $violations.Add("static NVRTC/HIPRTC link directive in $relativePath")
    }
}

if ($violations.Count -ne 0) {
    $detail = ($violations | Sort-Object -CaseSensitive -Unique) -join [Environment]::NewLine
    throw "SDK-free preflight failed:$([Environment]::NewLine)$detail"
}

# Runtime-only NVRTC/HIPRTC shared libraries are intentionally not matched.
# They are deployment capabilities loaded dynamically, not build inputs.
Write-Host 'SDK-free preflight passed: no toolkit compiler, configuration, header, import library, static archive/directive, or tracked vendor binary was found; untracked runtime-only shared libraries remain allowed.'
