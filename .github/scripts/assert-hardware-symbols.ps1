# SPDX-License-Identifier: CC0-1.0

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('cuda', 'hip')]
    [string] $Backend,

    [Parameter(Mandatory = $true)]
    [string] $SymbolsJson,

    [Parameter(Mandatory = $true)]
    [string] $GeneratedInventory
)

$ErrorActionPreference = 'Stop'
$caseProbe = @('ocgpuCaseProbe', 'ocgpuCASEProbe') | Sort-Object -CaseSensitive -Unique
if ($caseProbe.Count -ne 2 -or -not ('ocgpuCaseProbe' -cnotin @('ocgpuCASEProbe'))) {
    throw 'PowerShell case-sensitive symbol comparison invariant is unavailable'
}
$result = Get-Content -LiteralPath $SymbolsJson -Raw | ConvertFrom-Json
$generated = Get-Content -LiteralPath $GeneratedInventory -Raw | ConvertFrom-Json

if ($result.backend -ne $Backend) {
    throw "symbol report backend '$($result.backend)' does not match requested '$Backend'"
}
if ($result.library_error) {
    throw "$Backend runtime load failed: $($result.library_error)"
}

$expected = @(
    $generated.raw_inventory |
        Where-Object {
            $_.backend -eq $Backend -and
            $_.emitted -eq $true -and
            @($_.platforms) -contains $result.target
        } |
        ForEach-Object { $_.vendor_name } |
        Sort-Object -CaseSensitive -Unique
)
$platformName = if ($result.target -ceq 'x86_64-pc-windows-msvc') {
    'windows'
} elseif ($result.target -ceq 'x86_64-unknown-linux-gnu') {
    'linux'
} else {
    throw "unsupported hardware-smoke target '$($result.target)'"
}
$expectedCore = @(
    @($generated.function) |
        ForEach-Object {
            $mapping = $_.$Backend
            if ($null -ne $mapping -and $mapping.$platformName.available -eq $true) {
                if (-not [string]::IsNullOrWhiteSpace([string]$mapping.dispatch_symbol)) {
                    [string]$mapping.dispatch_symbol
                } else {
                    [string]$mapping.vendor_symbol
                }
            }
        } |
        Sort-Object -CaseSensitive -Unique
)
$reported = @($result.symbols)
$applicable = @($reported | Where-Object { $_.runtime_applicable -eq $true })
$observed = @($reported | ForEach-Object { $_.name } | Sort-Object -CaseSensitive -Unique)
$missingCoverage = @($expected | Where-Object { $_ -cnotin $observed })
$missingCore = @(
    foreach ($expectedName in $expectedCore) {
        $candidates = @($applicable | Where-Object { $_.name -ceq $expectedName })
        $supported = @(
            $candidates |
                Where-Object {
                    $_.runtime_required -eq $true -and
                    ($_.runtime_available -eq $true -or $_.runtime_resolution -ceq 'direct_adapter')
                }
        )
        if ($supported.Count -eq 0) {
            [pscustomobject]@{
                name = $expectedName
                runtime_resolution = @(
                    $candidates |
                        ForEach-Object { $_.runtime_resolution } |
                        Sort-Object -CaseSensitive -Unique
                ) -join ','
            }
        }
    }
)

if ($expected.Count -eq 0) {
    throw "generated inventory has no emitted $Backend symbols for target $($result.target)"
}
if ($expectedCore.Count -eq 0) {
    throw "generated inventory has no required $Backend common-core symbols for target $($result.target)"
}
if ($missingCoverage.Count -ne 0) {
    throw "coverage/runtime report omits $($missingCoverage.Count) emitted target symbols: $($missingCoverage -join ', ')"
}
if ($missingCore.Count -ne 0) {
    $details = $missingCore | ForEach-Object { "$($_.name) [$($_.runtime_resolution)]" }
    throw "$($missingCore.Count) required $Backend common-core symbols did not resolve through a direct, proc-address, or reviewed direct-adapter path: $($details -join ', ')"
}

$rawAvailable = @($applicable | Where-Object { $_.runtime_available -eq $true }).Count
$directAdapters = @($applicable | Where-Object { $_.runtime_resolution -ceq 'direct_adapter' }).Count
$profileOmissions = @($applicable | Where-Object { $_.runtime_resolution -ceq 'profile_unavailable' }).Count
$optionalMissing = @(
    $applicable |
        Where-Object {
            $_.runtime_required -ne $true -and $_.runtime_resolution -ceq 'missing'
        }
).Count
Write-Host "Validated $($expectedCore.Count) required $Backend common-core symbols for $($result.target); raw callable=$rawAvailable, direct adapters=$directAdapters, supported profile omissions=$profileOmissions, optional missing=$optionalMissing."
