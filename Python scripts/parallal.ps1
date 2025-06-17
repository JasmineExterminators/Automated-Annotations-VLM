<#
.SYNOPSIS

.PARAMETER PythonScript
Path to the Python script to run.

.PARAMETER ParentDir
Path to the parent directory containing subdirectories of videos.
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$PythonScript,

    [Parameter(Mandatory=$true)]
    [string]$ParentDir
)

# Prepare job list for cleanup
$jobs = @()


# Launch one Python process per subdirectory
$counter = 0
foreach ($dir in Get-ChildItem -Directory -Path $ParentDir) {
    $subdir = $dir.FullName
    # Start Python process
    $jobs += Start-Job -ScriptBlock { & python.exe $args[0] $args[1] } -ArgumentList $args[0], $subdir
}
    # Pin t

Wait-Job