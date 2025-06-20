<#
.SYNOPSIS
    Run one Python script per subdirectory in parallel (throttled to CPU count),
    then wait for them all and clean up.
.PARAMETER ParentDir
    Path to the parent directory containing subdirectories.
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$ParentDir
)

# how many parallel workers to allow
$maxConcurrent = [Environment]::ProcessorCount

# path to your Python script
$PythonScript = "C:\Users\wuad3\Documents\CMU\Freshman_Year\Research\Automated-Annotations-VLM\Python_scripts\auto_annotate_scripts\auto_annotate_final.py"

# collect all subdirs
$dirs = Get-ChildItem -Directory -Path $ParentDir

# array to hold our Job objects
$jobs = @()

foreach ($dir in $dirs) {
    # if we've reached our limit, wait for one to finish
    while (($jobs | Where-Object { $_.State -eq 'Running' }).Count -ge $maxConcurrent) {
        Start-Sleep -Milliseconds 200
    }

    # launch a new PSJob for this subdirectory
    $jobs += Start-Job -ScriptBlock {
        param($scriptPath, $dirPath)
        & python.exe $scriptPath $dirPath
    } -ArgumentList $PythonScript, $dir.FullName
}

# now wait for *all* jobs to finish
$jobs | Wait-Job

# inspect results / errors
foreach ($job in $jobs) {
    $child = $job.ChildJobs[0]
    if ($child.JobStateInfo.State -ne 'Completed') {
        Write-Host "❌ Job #$($job.Id) on '$($child.JobParameters[1])' failed: $($child.JobStateInfo.State)"
        if ($child.Error) {
            Write-Host "Errors:"
            $child.Error | ForEach-Object { "  $_" }
        }
    }
    else {
        Write-Host "✅ Job #$($job.Id) completed successfully."
    }
}

# (optional) grab the stdout from the very first job to demonstrate Receive-Job
Receive-Job -Job $jobs[0]

# clean up all jobs
$jobs | Remove-Job
