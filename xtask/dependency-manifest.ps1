# SPDX-License-Identifier: CC0-1.0

param(
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$metadataText = cargo metadata --locked --format-version 1 --manifest-path "$root/Cargo.toml"
if ($LASTEXITCODE -ne 0) { throw "cargo metadata failed with $LASTEXITCODE" }
$metadata = $metadataText | ConvertFrom-Json

# Sort-Object uses the current culture, whose punctuation collation differs
# between Windows NLS and Linux ICU. These manifests are byte-for-byte checked.
function Get-OrdinalSortedObjects([object[]]$Values, [scriptblock]$KeySelector) {
    $objectsByKey = [Collections.Generic.SortedDictionary[string, object]]::new(
        [StringComparer]::Ordinal
    )
    foreach ($value in $Values) {
        $key = [string](& $KeySelector $value)
        if ($objectsByKey.ContainsKey($key)) {
            throw "ordinal sort key is duplicated: $key"
        }
        $objectsByKey.Add($key, $value)
    }
    foreach ($entry in $objectsByKey.GetEnumerator()) { $entry.Value }
}

function Get-OrdinalSortedUniqueStrings([object[]]$Values) {
    $unique = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($value in $Values) { [void]$unique.Add([string]$value) }
    [string[]]$sorted = @($unique)
    [Array]::Sort($sorted, [StringComparer]::Ordinal)
    $sorted
}

$distributionRootNames = @(
    'ocgpu',
    'ocgpu-capi',
    'ocgpu-cli'
)
$nodesById = @{}
foreach ($node in $metadata.resolve.nodes) { $nodesById[[string]$node.id] = $node }
$shippingIds = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
$queue = [Collections.Generic.Queue[string]]::new()
foreach ($package in $metadata.packages | Where-Object { $_.name -in $distributionRootNames }) {
    $queue.Enqueue([string]$package.id)
}
while ($queue.Count -gt 0) {
    $id = $queue.Dequeue()
    if (-not $shippingIds.Add($id)) { continue }
    $node = $nodesById[$id]
    if ($null -eq $node) { continue }
    foreach ($dependency in $node.deps) {
        $nonDevelopmentKinds = @($dependency.dep_kinds | Where-Object { $_.kind -ne 'dev' })
        if ($nonDevelopmentKinds.Count -gt 0) { $queue.Enqueue([string]$dependency.pkg) }
    }
}

$lockText = Get-Content -Raw "$root/Cargo.lock"
$checksums = @{}
foreach ($match in [regex]::Matches($lockText, '(?ms)^\[\[package\]\]\r?\n(.*?)(?=^\[\[package\]\]|\z)')) {
    $body = $match.Groups[1].Value
    $name = [regex]::Match($body, '(?m)^name = "([^"]+)"\r?$').Groups[1].Value
    $version = [regex]::Match($body, '(?m)^version = "([^"]+)"\r?$').Groups[1].Value
    $checksum = [regex]::Match($body, '(?m)^checksum = "([0-9a-f]{64})"\r?$').Groups[1].Value
    if ($name -and $version -and $checksum) { $checksums["$name@$version"] = $checksum }
}

$packages = @(Get-OrdinalSortedObjects @($metadata.packages) {
    param($package)
    "{0}`0{1}`0{2}" -f $package.name, $package.version, $package.id
} | ForEach-Object {
    $key = "$($_.name)@$($_.version)"
    $checksum = $checksums[$key]
    $download = if ($_.source -like 'registry+*') {
        "https://crates.io/api/v1/crates/$($_.name)/$($_.version)/download"
    } else { $null }
    [ordered]@{
        name = $_.name
        version = $_.version
        license_expression = $_.license
        source = $_.source
        checksum_sha256 = $checksum
        repository = $_.repository
        download = $download
        scope = if ($shippingIds.Contains([string]$_.id)) {
            'shipping_runtime_or_build'
        } else {
            'repository_development_only'
        }
    }
})

$manifest = [ordered]@{
    schema_version = 1
    spdx_license_identifier = 'CC0-1.0'
    document_license_scope = 'The SPDX identifier licenses this independently authored inventory document, not the listed packages.'
    generated_from = 'Cargo.lock and cargo metadata --locked --format-version 1'
    shipping_roots = $distributionRootNames
    packages = $packages
}

$refs = @{}
foreach ($package in $metadata.packages) {
    $refs[[string]$package.id] = "pkg:cargo/$($package.name)@$($package.version)"
}
$shippingPackages = @($metadata.packages |
    Where-Object { $shippingIds.Contains([string]$_.id) })
$components = @(Get-OrdinalSortedObjects $shippingPackages {
    param($package)
    "{0}`0{1}`0{2}" -f $package.name, $package.version, $package.id
} |
    ForEach-Object {
        $licenseExpression = [string]$_.license
        if ([string]::IsNullOrWhiteSpace($licenseExpression)) {
            throw "shipping component $($_.name)@$($_.version) has no license expression"
        }
        $licenseChoices = [Collections.Generic.List[object]]::new()
        $licenseChoices.Add([ordered]@{ expression = $licenseExpression })
        $component = [ordered]@{
            type = if ($_.name -ceq 'ocgpu-cli') { 'application' } else { 'library' }
            'bom-ref' = $refs[[string]$_.id]
            name = $_.name
            version = $_.version
            purl = $refs[[string]$_.id]
            licenses = $licenseChoices
        }
        $checksum = $checksums["$($_.name)@$($_.version)"]
        if ($checksum) { $component.hashes = @([ordered]@{ alg = 'SHA-256'; content = $checksum }) }
        if ($_.repository) {
            $component.externalReferences = @([ordered]@{ type = 'vcs'; url = $_.repository })
        }
        $component
    })
$shippingNodes = @($metadata.resolve.nodes |
    Where-Object { $shippingIds.Contains([string]$_.id) })
$componentDependencies = @(Get-OrdinalSortedObjects $shippingNodes {
    param($node)
    [string]$node.id
} |
    ForEach-Object {
        $node = $_
        $dependencyRefs = @($node.deps |
            Where-Object {
                $_.pkg -and
                $shippingIds.Contains([string]$_.pkg) -and
                @($_.dep_kinds | Where-Object { $_.kind -ne 'dev' }).Count -gt 0
            } |
            ForEach-Object { $refs[[string]$_.pkg] })
        $dependsOn = @(Get-OrdinalSortedUniqueStrings $dependencyRefs)
        [ordered]@{
            ref = $refs[[string]$node.id]
            dependsOn = $dependsOn
        }
    })
$rootPackage = $metadata.packages | Where-Object { $_.name -ceq 'ocgpu' } | Select-Object -First 1
if ($null -eq $rootPackage) { throw 'shipping root package ocgpu is absent from cargo metadata' }
$distributionRootRefs = @($distributionRootNames | ForEach-Object {
    $rootName = $_
    $matches = @($metadata.packages | Where-Object { $_.name -ceq $rootName })
    if ($matches.Count -ne 1) { throw "distribution root $rootName must identify exactly one package" }
    $refs[[string]$matches[0].id]
})
$aggregateRef = "urn:ocgpu:distribution:$($rootPackage.version)"
$dependencies = @(
    [ordered]@{
        ref = $aggregateRef
        dependsOn = $distributionRootRefs
    }
) + $componentDependencies
$sbom = [ordered]@{
    bomFormat = 'CycloneDX'
    specVersion = '1.5'
    version = 1
    metadata = [ordered]@{
        properties = @(
            [ordered]@{ name = 'ocgpu:document-license'; value = 'CC0-1.0' },
            [ordered]@{ name = 'ocgpu:component-scope'; value = 'shipping-runtime-and-build-reachable; Cargo dev dependencies excluded' },
            [ordered]@{ name = 'ocgpu:repository-development-inventory'; value = 'LICENSES/dependencies.json' }
        )
        component = [ordered]@{
            type = 'application'
            'bom-ref' = $aggregateRef
            name = 'ocgpu distribution'
            version = $rootPackage.version
            licenses = @([ordered]@{ expression = 'CC0-1.0' })
        }
    }
    components = $components
    dependencies = $dependencies
}

function Test-CycloneDxShape([object]$Bom, [string]$ExpectedAggregateRef, [string[]]$ExpectedRootRefs) {
    if ($Bom.bomFormat -ne 'CycloneDX' -or $Bom.specVersion -ne '1.5' -or $Bom.version -ne 1) {
        throw 'CycloneDX identity fields are invalid'
    }
    $componentRefs = @($Bom.components | ForEach-Object { $_.'bom-ref' })
    if ($componentRefs.Count -eq 0 -or
        @(Get-OrdinalSortedUniqueStrings $componentRefs).Count -ne $componentRefs.Count) {
        throw 'CycloneDX components must have unique non-empty references'
    }
    if ($Bom.metadata.component.'bom-ref' -cne $ExpectedAggregateRef) {
        throw 'CycloneDX metadata component is not the expected aggregate product'
    }
    if (@($componentRefs | Where-Object { $_ -ceq $ExpectedAggregateRef }).Count -ne 0) {
        throw 'CycloneDX aggregate product must not duplicate a package component reference'
    }

    $knownRefs = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    [void]$knownRefs.Add($ExpectedAggregateRef)
    foreach ($componentRef in $componentRefs) { [void]$knownRefs.Add([string]$componentRef) }
    $dependencyRefs = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    foreach ($dependency in $Bom.dependencies) {
        if (-not $knownRefs.Contains([string]$dependency.ref)) {
            throw "CycloneDX dependency ref $($dependency.ref) is absent"
        }
        if (-not $dependencyRefs.Add([string]$dependency.ref)) {
            throw "CycloneDX dependency ref $($dependency.ref) is duplicated"
        }
        foreach ($target in $dependency.dependsOn) {
            if (-not $knownRefs.Contains([string]$target)) {
                throw "CycloneDX dependency target $target is absent"
            }
            if ([string]$target -ceq [string]$dependency.ref) {
                throw "CycloneDX dependency ref $($dependency.ref) depends on itself"
            }
        }
    }

    if ($dependencyRefs.Count -ne $knownRefs.Count) {
        throw 'CycloneDX must contain exactly one dependency node for the aggregate and every component'
    }
    $aggregateEdges = @($Bom.dependencies | Where-Object { $_.ref -ceq $ExpectedAggregateRef })
    if ($aggregateEdges.Count -ne 1) { throw 'CycloneDX aggregate dependency node is missing or duplicated' }
    $actualRootRefs = @(Get-OrdinalSortedUniqueStrings @($aggregateEdges[0].dependsOn))
    $expectedRootRefsSorted = @(Get-OrdinalSortedUniqueStrings $ExpectedRootRefs)
    if ($actualRootRefs.Count -ne $expectedRootRefsSorted.Count -or
        [string]::Join("`n", $actualRootRefs) -cne [string]::Join("`n", $expectedRootRefsSorted)) {
        throw 'CycloneDX aggregate dependency node does not point to the exact distribution roots'
    }

    $dependencyMap = @{}
    foreach ($dependency in $Bom.dependencies) { $dependencyMap[[string]$dependency.ref] = @($dependency.dependsOn) }
    $reachable = [Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
    $queue = [Collections.Generic.Queue[string]]::new()
    $queue.Enqueue($ExpectedAggregateRef)
    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        if (-not $reachable.Add($current)) { continue }
        foreach ($target in $dependencyMap[$current]) { $queue.Enqueue([string]$target) }
    }
    if ($reachable.Count -ne $knownRefs.Count) {
        throw 'CycloneDX contains a component not reachable from the aggregate product'
    }
}

function Test-CycloneDxLicenseArray([object]$Component, [string]$Label) {
    $licensesProperty = $Component.PSObject.Properties['licenses']
    if ($null -eq $licensesProperty -or -not ($licensesProperty.Value -is [array])) {
        throw "CycloneDX $Label licenses must serialize as an array"
    }
    $choices = @($licensesProperty.Value)
    if ($choices.Count -eq 0) { throw "CycloneDX $Label licenses array is empty" }
    foreach ($choice in $choices) {
        $propertyNames = @($choice.PSObject.Properties | ForEach-Object { $_.Name })
        $hasExpression = $propertyNames -ccontains 'expression'
        $hasLicense = $propertyNames -ccontains 'license'
        if ($hasExpression -eq $hasLicense) {
            throw "CycloneDX $Label license choice must contain exactly one of expression or license"
        }
        if ($hasExpression) {
            $expression = $choice.PSObject.Properties['expression'].Value
            if (-not ($expression -is [string]) -or [string]::IsNullOrWhiteSpace($expression)) {
                throw "CycloneDX $Label license expression must be a nonempty string"
            }
            continue
        }

        $license = $choice.PSObject.Properties['license'].Value
        if ($null -eq $license) { throw "CycloneDX $Label license object is null" }
        $licenseProperties = @($license.PSObject.Properties | ForEach-Object { $_.Name })
        $hasId = $licenseProperties -ccontains 'id'
        $hasName = $licenseProperties -ccontains 'name'
        if ($hasId -eq $hasName) {
            throw "CycloneDX $Label license object must contain exactly one of id or name"
        }
        $identity = if ($hasId) {
            $license.PSObject.Properties['id'].Value
        } else {
            $license.PSObject.Properties['name'].Value
        }
        if (-not ($identity -is [string]) -or [string]::IsNullOrWhiteSpace($identity)) {
            throw "CycloneDX $Label license identity must be a nonempty string"
        }
    }
}

function Test-CycloneDxSerializedLicenses([string]$Content) {
    $parsed = $Content | ConvertFrom-Json
    Test-CycloneDxLicenseArray $parsed.metadata.component 'metadata component'
    foreach ($component in $parsed.components) {
        Test-CycloneDxLicenseArray $component "component $($component.'bom-ref')"
    }
}

function Write-Or-Check([string]$Path, [object]$Value) {
    $content = ($Value | ConvertTo-Json -Depth 20 -Compress) + "`n"
    if ($Check) {
        if (-not (Test-Path -LiteralPath $Path)) { throw "$Path is missing" }
        $actual = [System.IO.File]::ReadAllText($Path)
        if ($actual -ne $content) { throw "$Path is stale; run cargo run -p xtask -- generate" }
    } else {
        [System.IO.File]::WriteAllText($Path, $content, [System.Text.UTF8Encoding]::new($false))
        Write-Host "wrote $Path"
    }
}

Test-CycloneDxShape $sbom $aggregateRef $distributionRootRefs
$serializedSbom = ($sbom | ConvertTo-Json -Depth 20 -Compress) + "`n"
Test-CycloneDxSerializedLicenses $serializedSbom
Write-Or-Check "$root/LICENSES/dependencies.json" $manifest
Write-Or-Check "$root/LICENSES/ocgpu.cdx.json" $sbom
