# Troubleshooting

## Common Issues

### Installation Problems

**CUDA Version Mismatch**
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Reinstall with correct CUDA version
pip install -r requirements.txt
```

## Debugging Tips

### Debug Build

For debugging C++ code issues, build with debug symbols enabled:

```bash
# Build with debug symbols (includes debugging information)
make build/native BUILD_TYPE=Debug

# For incremental builds during development
make build/native_incremental
```

**Debug vs Release builds:**
- `Debug` (default): Includes debug symbols, no optimization, easier debugging
- `Release`: Optimized for performance, smaller binaries, harder to debug

### Enable Debug Logs

Control logging verbosity using the `SETU_LOG_LEVEL` environment variable:

```bash
# Enable debug logging (most verbose)
export SETU_LOG_LEVEL=DEBUG
python your_script.py

# Other log levels
export SETU_LOG_LEVEL=INFO     # Default level
export SETU_LOG_LEVEL=WARNING  # Warnings and errors only
export SETU_LOG_LEVEL=ERROR    # Errors only
```

**Understanding Log Format:**

Setu logs follow a structured format that includes process identification and timing information:

```
YYMMDD HH:MM:SS.uuuuuu PID:TID PROCESS_ID:THREAD_NAME] [LOG_LEVEL] [FILE:LINE] MESSAGE
```

Where:
- `YYMMDD HH:MM:SS.uuuuuu`: Timestamp (year/month/day hour:minute:second.microseconds)
- `PID:TID`: Process ID and Thread ID
- `PROCESS_ID`: Process type identifier:
  - `M`: Main process
  - `C-{replica_id}`: Controller process for given replica
  - `W-{replica_id}-{rank}`: Worker process for given replica and rank
- `THREAD_NAME`: Thread name (e.g., "SchedulerLoop-0", "ExecutionLoop-1")
- `LOG_LEVEL`: Log severity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `FILE:LINE`: Source file and line number
- `MESSAGE`: Actual log message

**Example log entries:**
```
250613 17:29:49.737246 3662251:8f2a1 W-0-1:ExecutionLoop-1] [INFO] [BaseWorker.cpp:91] BaseWorker 1 initialized
250613 17:29:49.737412 3662251:8f2a2 C-0:SchedulerLoop-0] [DEBUG] [BaseReplicaController.cpp:298] Starting scheduler step
250613 17:29:49.737503 3662251:8f2a3 M:main] [ERROR] [InferenceEngine.cpp:42] Failed to initialize engine
```

### Environment Diagnostics

**Collect environment information:**
```bash
# Print comprehensive environment info
python -m setu.utils.collect_env
```

### Memory Issues

**Reduce build memory usage:**

```bash
# Limit parallel build jobs
export CMAKE_BUILD_PARALLEL_LEVEL=4

make build/native
```
