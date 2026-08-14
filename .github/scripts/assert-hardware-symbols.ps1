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
$applicable = @($result.symbols | Where-Object { $_.runtime_applicable -eq $true })
$observed = @($applicable | ForEach-Object { $_.name } | Sort-Object -CaseSensitive -Unique)
$missingCoverage = @($expected | Where-Object { $_ -cnotin $observed })
$missingRuntime = @(
    foreach ($expectedName in $expected) {
        $candidates = @($applicable | Where-Object { $_.name -ceq $expectedName })
        if (-not ($candidates | Where-Object { $_.runtime_available -eq $true })) {
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
if ($missingCoverage.Count -ne 0) {
    throw "coverage/runtime report omits $($missingCoverage.Count) emitted target-applicable symbols: $($missingCoverage -join ', ')"
}
if ($missingRuntime.Count -ne 0) {
    $details = $missingRuntime | ForEach-Object { "$($_.name) [$($_.runtime_resolution)]" }
    throw "$($missingRuntime.Count) target-applicable $Backend symbols did not resolve: $($details -join ', ')"
}

Write-Host "Resolved all $($expected.Count) emitted $Backend symbols applicable to $($result.target)."
