import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { apiRouter } from './server/src/routes/api';

dotenv.config();

async function startServer() {
  const app = express();
  const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
  const isProd = process.env.NODE_ENV === 'production';

  // Body parsing for JSON images (allow up to 50MB for high-res camera captures)
  app.use(express.json({ limit: '50mb' }));
  app.use(express.urlencoded({ extended: true, limit: '50mb' }));

  // Mount backend API routes
  app.use('/api', apiRouter);

  // Guarantee /api/* never falls through to Vite SPA HTML fallback
  app.use('/api', (_req, res) => {
    res.status(404).json({ success: false, error: 'API endpoint not found.' });
  });

  // Guarantee /api/* middleware errors (e.g. PayloadTooLargeError, malformed JSON) return JSON
  app.use(
    '/api',
    (
      err: any,
      _req: express.Request,
      res: express.Response,
      _next: express.NextFunction
    ) => {
      const status = err?.status || err?.statusCode || 500;
      const message =
        err?.type === 'entity.too.large'
          ? 'Uploaded image is too large. Please select a smaller image or retake photo.'
          : err?.message || 'Unexpected API server error.';
      res.status(status).json({ success: false, error: message });
    }
  );

  if (!isProd) {
    // Development mode with Vite middleware
    const { createServer } = await import('vite');
    const vite = await createServer({
      server: {
        middlewareMode: true,
        hmr: process.env.DISABLE_HMR !== 'true',
      },
      appType: 'spa',
    });

    app.use(vite.middlewares);
  } else {
    // Production static serving
    const distPath = path.resolve(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      if (req.path.startsWith('/api')) {
        return res.status(404).json({ success: false, error: 'API endpoint not found.' });
      }
      return res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`[AgriGrade Server] Running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('[AgriGrade Server] Fatal startup error:', err);
  process.exit(1);
});
