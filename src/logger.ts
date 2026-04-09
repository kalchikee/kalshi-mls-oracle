// MLS Oracle v4.1 — Logger (pino)

import pino from 'pino';
import { mkdirSync } from 'fs';
import { resolve } from 'path';

const LOG_DIR = resolve('./logs');
mkdirSync(LOG_DIR, { recursive: true });

const isDev = process.env.NODE_ENV !== 'production';

export const logger = pino(
  {
    level: process.env.LOG_LEVEL ?? 'info',
    base: { pid: process.pid },
    timestamp: pino.stdTimeFunctions.isoTime,
  },
  isDev
    ? pino.transport({
        target: 'pino-pretty',
        options: {
          colorize: true,
          translateTime: 'HH:MM:ss',
          ignore: 'pid,hostname',
        },
      })
    : pino.destination(resolve(LOG_DIR, 'mls_oracle.log'))
);
