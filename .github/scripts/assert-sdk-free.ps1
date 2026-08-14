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
        "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\*\lib\x64\cuda.lib",
        "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\*\lib\x64\cudart.lib",
        "$env:ProgramFiles\AMD\ROCm\*\include\hip\hip_runtime_api.h",
        "$env:ProgramFiles\AMD\ROCm\*\lib\amdhip64.lib",
        "$env:ProgramFiles\AMD\ROCm\*\lib\hiprtc.lib"
    )
} else {
    @(
        '/opt/cuda/include/cuda.h',
        '/opt/cuda/targets/*/include/cuda.h',
        '/opt/rocm/include/hip/hip_runtime_api.h',
        '/opt/rocm/lib/cmake/hip/hip-config.cmake',
        '/opt/rocm-*/include/hip/hip_runtime_api.h',
        '/opt/rocm-*/lib/cmake/hip/hip-config.cmake',
        '/usr/include/cuda.h',
        '/usr/include/hip/hip_runtime_api.h',
        '/usr/local/cuda/include/cuda.h',
        '/usr/local/cuda/lib64/stubs/libcuda.so',
        '/usr/local/cuda/targets/*/include/cuda.h',
        '/usr/local/include/hip/hip_runtime_api.h'
    )
}
foreach ($pattern in $sdkArtifactPatterns) {
    foreach ($match in @(Get-Item -Path $pattern -ErrorAction SilentlyContinue)) {
        $violations.Add("SDK header/link-time artifact $($match.FullName)")
    }
}

if ($violations.Count -ne 0) {
    $detail = ($violations | Sort-Object -CaseSensitive -Unique) -join [Environment]::NewLine
    throw "SDK-free preflight failed:$([Environment]::NewLine)$detail"
}

Write-Host 'SDK-free preflight passed: no toolkit compiler, configuration, header, or link-time artifact was found.'
