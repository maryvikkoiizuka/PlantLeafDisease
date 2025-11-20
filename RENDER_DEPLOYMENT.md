# Render.com Deployment Guide

This guide walks you through deploying the Plant Leaf Disease Prediction app to Render.com.

## Prerequisites

- A [Render.com](https://render.com) account
- This repository pushed to GitHub (public or private)
- Git installed locally

## Deployment Steps

### 1. Generate a Secure Secret Key

Generate a new SECRET_KEY for production:

```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

Copy the output for later use.

### 2. Create a Render.com Web Service

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Select **Build and deploy from a Git repository**
4. Connect your GitHub repository
5. Configure the following settings:

#### Basic Settings
- **Name**: `plant-leaf-disease-prediction`
- **Environment**: `Python`
- **Region**: Choose closest to your users
- **Branch**: `main`

#### Build Command
Leave default or use:
```bash
pip install -r requirements.txt && python manage.py collectstatic --noinput
```

#### Start Command
```bash
bash ./scripts/entrypoint.sh
```

#### Environment Variables (click "Add Environment Variable")

Add the following variables:

| Key | Value | Notes |
|-----|-------|-------|
| `DEBUG` | `False` | Always False for production |
| `SECRET_KEY` | `<your-generated-key>` | Use the key generated in step 1 |
| `ALLOWED_HOSTS` | `<service-name>.onrender.com,www.<service-name>.onrender.com` | Replace with your actual domain |
| `TF_CPP_MIN_LOG_LEVEL` | `2` | Reduce TensorFlow logging |
| `OMP_NUM_THREADS` | `1` | Optimize for limited resources |

**Optional Performance Tuning:**
- `TF_NUM_INTRAOP_THREADS`: `1`
- `TF_NUM_INTEROP_THREADS`: `1`
- `GUNICORN_CMD_ARGS`: `--workers=1 --threads=2 --timeout 120`

### 3. Resource Configuration

- **Instance Type**: Standard (recommended minimum for ML models)
- **Auto-deploy**: Enable for automatic redeploy on git push
- **Persistent Disk**: Optional (for storing uploaded images)

### 4. Deploy

Click **Create Web Service** and wait for deployment to complete (may take 5-10 minutes).

## Post-Deployment

### Verify Deployment

1. Check logs in Render dashboard for any errors
2. Visit your app URL to verify it's running
3. Test the image upload and analysis features

### Monitor Performance

- Watch TensorFlow memory usage in logs
- If out-of-memory errors occur, upgrade instance type
- Check response times for model predictions

### Update Models

If you want to update the ML model:

1. Update model files in the `/models` directory
2. Push to GitHub
3. Render will automatically rebuild and redeploy

## Troubleshooting

### Port Bind Errors
The entrypoint script includes retry logic for port conflicts. If still failing:
- Check logs for specific error message
- Increase `WAIT_FOR_PORT_FREE_SECONDS` environment variable (default: 120)

### Out of Memory Errors
- Upgrade to a larger instance
- Reduce `GUNICORN_CMD_ARGS` workers/threads
- Consider using TensorFlow Serving for better memory management

### Static Files Not Loading
- Verify `collectstatic` runs successfully in build logs
- Ensure `STATIC_ROOT` and `STATIC_URL` are configured correctly
- WhiteNoise middleware is already configured

### Model Loading Fails
- Check that model files exist in `/models` directory
- Verify model file format matches expected `.keras` format
- Check logs for TensorFlow initialization errors

## Security Recommendations

1. **Never commit `.env` file** - use `.env.example` instead
2. **Use strong SECRET_KEY** - already configured to use environment variable
3. **Enable HTTPS** - automatically handled by Render
4. **Keep dependencies updated** - regularly update `requirements.txt`
5. **Monitor logs** - review for suspicious activity

## Custom Domain

To use a custom domain:

1. In Render dashboard, go to your service
2. Navigate to **Settings** → **Custom Domains**
3. Add your domain and follow DNS setup instructions
4. Update `ALLOWED_HOSTS` in environment variables

## Performance Optimization Tips

### For ML Models on Limited Resources:
- Use TensorFlow Lite quantized models for faster inference
- Consider model compression techniques
- Cache predictions if possible
- Use a CDN for static files (Render can handle this)

### Instance Selection:
- **Starter** ($0): Limited, suitable for testing only
- **Standard** ($7-15): Recommended minimum for production
- **Pro** ($25+): For higher traffic and larger models

## Estimated Costs

- **Compute**: $7-15/month (Standard instance)
- **Bandwidth**: Free up to 100GB/month, then $0.10/GB
- **Storage**: $10/month (optional, for persistent storage)

**Total**: ~$17-25/month for production-ready deployment

## Need Help?

- Check [Render Documentation](https://render.com/docs)
- Review logs in Render dashboard
- Check Django logs for application-specific issues
- Visit [TensorFlow Deployment Guide](https://www.tensorflow.org/guide/deployment)
