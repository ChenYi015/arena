## arena serve logs

Print the logs of a serving job

### Synopsis

Print the logs of a serving job

```
arena serve logs JOB [-T JOB_TYPE] [-v JOB_VERSION] [-i JOB_INSTANCE] [flags]
```

### Options

```
  -c, --container string    Specify the container name of instance to get log
  -f, --follow              Specify if the logs should be streamed.
  -h, --help                help for logs
  -i, --instance string     Specify the task instance to get log
      --since string        Only return logs newer than a relative duration like 5s, 2m, or 3h. Defaults to all logs. Only one of since-time / since may be used.
      --since-time string   Only return logs after a specific date (RFC3339). Defaults to all logs. Only one of since-time / since may be used.
  -t, --tail int            Lines of recent log file to display. Defaults to -1 with no selector, showing all log lines otherwise 10, if a selector is provided. (default -1)
      --timestamps          Include timestamps on each line in the log output
  -T, --type string         The serving type, the possible option is [tf(Tensorflow),trt(Tensorrt),seldon(Seldon),custom(Custom),kf(KFServing)]. (optional)
  -v, --version string      set the serving job version
```

### Options inherited from parent commands

```
      --arena-namespace string   The namespace of arena system service, like tf-operator (default "arena-system")
      --config string            Path to a kube config. Only required if out-of-cluster
      --loglevel string          Set the logging level. One of: debug|info|warn|error (default "info")
  -n, --namespace string         the namespace of the job
      --pprof                    enable cpu profile
      --trace                    enable trace
```

### SEE ALSO

* [arena serve](arena_serve.md)	 - Serve a job.

###### Auto generated by spf13/cobra on 5-Mar-2021