<#
.SYNOPSIS
    Run one Python script per subdirectory in parallel,
    then wait for them all and clean up.
.PARAMETER PythonScript
    Path to the Python script to run.
.PARAMETER ParentDir
    Path to the parent directory containing subdirectories.
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$PythonScript,

    [Parameter(Mandatory=$true)]
    [string]$ParentDir
)


# 1) Start one background job per subdirectory
$jobs = Get-ChildItem -Directory -Path $ParentDir | ForEach-Object {
    Start-Job -ScriptBlock {
        param($scriptPath, $dirPath)
        & python.exe $scriptPath $dirPath
    } -ArgumentList $PythonScript, $_.FullName
}

# 2) Block until *those* jobs complete
# Wait for every job to finish
Wait-Job -Job $jobs

# Check each job for errors
foreach ($job in $jobs) {
    # The child job’s final state
    $state = $job.ChildJobs[0].JobStateInfo.State

    if ($state -ne 'Completed') {
        Write-Host "❌ Job #$($job.Id) for directory $($job.ChildJobs[0].JobParameters[1]) failed with state: $state"

        # Pull out its error records
        $errs = $job.ChildJobs[0].Error
        if ($errs) {
            "Errors from job:" 
            $errs | ForEach-Object { "  $_" }
        }
    }
    else {
        Write-Host "✅ Job #$($job.Id) completed successfully."
    }
}

# Clean up
$jobs | Remove-Job

