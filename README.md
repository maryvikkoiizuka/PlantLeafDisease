# Plant Leaf Disease Prediction

This repository contains a Django app that serves a Keras/TensorFlow plant disease classifier.

## Render deployment

If you deploy this project to Render.com, ensure Render starts exactly one process using the provided entrypoint script. Add the following Start Command in the Render Web Service settings (or include a `Procfile` in the repo):

```bash
bash ./scripts/entrypoint.sh
```

Notes and troubleshooting:

- The entrypoint acquires a file lock so only one process runs migrations, collects static files, and binds the configured `$PORT`. This avoids "Connection in use" errors caused by multiple start commands.
- If you still see "Connection in use" in the logs, check the entrypoint output — it prints the PID and command that currently owns the port. That helps identify duplicate start commands or crash-looping processes.
- If TensorFlow runs out of memory on Render, try a larger instance, or use Docker to control the Python runtime and TF wheel more precisely, or serve the model via TensorFlow Serving.

After updating the Start Command or adding the `Procfile`, redeploy the service from the Render dashboard.
## Port wait configuration

The entrypoint script waits briefly for Render's port detector to release the assigned `$PORT` before starting Gunicorn. If your service occasionally races with Render's internal port checker you can increase the wait timeout by setting an environment variable in the Render dashboard:

```bash
WAIT_FOR_PORT_FREE_SECONDS=60
```

Set it higher (for example `120`) if you continue to see port races. The default in the repo is 60 seconds.

