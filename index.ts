import 'dotenv/config';
import { GoogleGenAI } from '@google/genai';
import { ChromaClient } from 'chromadb';
import fs from 'fs/promises';

// ─── Client Initialization ───────────────────────────────────────────────────

const ai = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY as string });
const chroma = new ChromaClient({
  host: process.env.CHROMA_HOST as string || 'localhost',
  port: parseInt(process.env.CHROMA_PORT as string, 10) || 8000,
});

const EMBEDDING_MODEL = 'gemini-embedding-2-preview';

// ─── Types ───────────────────────────────────────────────────────────────────

type ImageMime = 'image/jpeg' | 'image/png' | 'image/webp' | 'image/gif';
type VideoMime = 'video/mp4' | 'video/webm';
type AudioMime = 'audio/mp3' | 'audio/wav' | 'audio/ogg';
type FileMime  = 'application/pdf' | 'text/plain';
type MimeType  = ImageMime | VideoMime | AudioMime | FileMime;

interface StoreOptions {
  collection: string;
  ids: string[];
  metadatas?: Record<string, string | number | boolean>[];
}

// ─── Core Helpers ────────────────────────────────────────────────────────────

/** Converts a list of content objects into embedding vectors. */
async function embedContents(contents: object[]): Promise<number[][]> {
  return Promise.all(
    contents.map(async (content) => {
      const response = await ai.models.embedContent({
        model: EMBEDDING_MODEL,
        contents: [content],
      });
      const values = response.embeddings?.[0]?.values;
      if (!values) throw new Error('No embedding returned from the API.');
      return values;
    })
  );
}

/** Converts a buffer to a base64 inline part (image / video / audio). */
function toInlinePart(buffer: Buffer, mimeType: MimeType) {
  return { inlineData: { data: buffer.toString('base64'), mimeType } };
}

/**
 * Converts a buffer to a text or PDF part.
 * - text/plain      → sent as plain text (saves tokens).
 * - application/pdf → sent as base64 inline data.
 */
function toFilePart(buffer: Buffer, mimeType: FileMime) {
  if (mimeType === 'text/plain') return { text: buffer.toString('utf-8') };
  return toInlinePart(buffer, mimeType);
}

/** Upserts embeddings into a Chroma collection. */
async function upsertToChroma(
  opts: StoreOptions,
  embeddings: number[][],
  documents: string[]
): Promise<void> {
  const collection = await chroma.getOrCreateCollection({ name: opts.collection });
  await collection.upsert({
    ids: opts.ids,
    embeddings,
    documents,
    metadatas: opts.metadatas,
  });
}

// ─── Embedding Functions ─────────────────────────────────────────────────────

export const embedTexts  = (texts: string[]) =>
  embedContents(texts.map((t) => ({ text: t })));

export const embedImages = (buffers: Buffer[], mime: ImageMime = 'image/jpeg') =>
  embedContents(buffers.map((b) => toInlinePart(b, mime)));

export const embedVideos = (buffers: Buffer[], mime: VideoMime = 'video/mp4') =>
  embedContents(buffers.map((b) => toInlinePart(b, mime)));

export const embedAudios = (buffers: Buffer[], mime: AudioMime = 'audio/mp3') =>
  embedContents(buffers.map((b) => toInlinePart(b, mime)));

export const embedFiles  = (buffers: Buffer[], mime: FileMime = 'application/pdf') =>
  embedContents(buffers.map((b) => toFilePart(b, mime)));

// ─── Store Functions ─────────────────────────────────────────────────────────
// All store functions accept pre-computed embeddings — no re-embedding is done here.

export async function storeTexts(texts: string[], embeddings: number[][], opts: StoreOptions) {
  await upsertToChroma(opts, embeddings, texts);
}

export async function storeImages(embeddings: number[][], opts: StoreOptions) {
  await upsertToChroma(opts, embeddings, opts.ids.map((id) => `image:${id}`));
}

export async function storeVideos(embeddings: number[][], opts: StoreOptions) {
  await upsertToChroma(opts, embeddings, opts.ids.map((id) => `video:${id}`));
}

export async function storeAudios(embeddings: number[][], opts: StoreOptions) {
  await upsertToChroma(opts, embeddings, opts.ids.map((id) => `audio:${id}`));
}

export async function storeFiles(embeddings: number[][], opts: StoreOptions) {
  await upsertToChroma(opts, embeddings, opts.ids.map((id) => `file:${id}`));
}

// ─── Test Utilities ──────────────────────────────────────────────────────────

/** Prints query results in a structured, readable format. */
function printQueryResults(
  label: string,
  query: string,
  results: { ids: string[][]; documents: (string | null)[][]; metadatas: (Record<string, unknown> | null)[][] }
) {
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`📌 ${label}`);
  console.log(`🔍 Query : "${query}"`);
  console.log(`${'─'.repeat(60)}`);

  results.ids[0].forEach((id, i) => {
    const doc      = results.documents?.[0]?.[i] ?? '—';
    const meta     = results.metadatas?.[0]?.[i];
    const distance = (results as any).distances?.[0]?.[i];
    console.log(
      `  #${i + 1}  id=${id}` +
      (distance !== undefined ? `  dist=${distance.toFixed(4)}` : '') +
      `\n       doc="${doc}"` +
      (meta ? `\n       meta=${JSON.stringify(meta)}` : '')
    );
  });
}

/** Prints the vector dimension and the first 4 values of the first embedding. */
function printEmbeddingStats(label: string, embeddings: number[][]) {
  console.log(`\n✅ ${label} — ${embeddings.length} vector(s), dim=${embeddings[0].length}`);
  console.log(`   First 4 values: [${embeddings[0].slice(0, 4).map((v) => v.toFixed(6)).join(', ')} …]`);
}

// ─── Test Examples ───────────────────────────────────────────────────────────

async function textExample() {
  console.log('\n═══════════════════ TEXT TEST ═══════════════════');

  const COLLECTION = 'test_texts';
  const texts = [
    'Allahu Akbar means "God is the greatest" in Arabic.',
    'The universe is vast and full of mysteries.',
    'Artificial intelligence is transforming the world.',
    'The quick brown fox jumps over the lazy dog.',
  ];
  const ids = ['text_1', 'text_2', 'text_3', 'text_4'];
  const metadatas = [
    { category: 'religion',   language: 'en' },
    { category: 'science',    language: 'en' },
    { category: 'technology', language: 'en' },
    { category: 'general',    language: 'en' },
  ];

  const embeddings = await embedTexts(texts);
  printEmbeddingStats('Text embeddings', embeddings);

  await storeTexts(texts, embeddings, { collection: COLLECTION, ids, metadatas });

  const collection = await chroma.getOrCreateCollection({ name: COLLECTION });

  // Query 1: expects exact semantic match
  const q1 = 'What jumps over the lazy dog?';
  const [qEmbed1] = await embedTexts([q1]);
  const r1 = await collection.query({ queryEmbeddings: [qEmbed1], nResults: 2, include: ['documents', 'metadatas', 'distances'] as any });
  printQueryResults('Text Query 1 — expected: text_4', q1, r1);

  // Query 2: expects semantic proximity
  const q2 = 'AI and machine learning';
  const [qEmbed2] = await embedTexts([q2]);
  const r2 = await collection.query({ queryEmbeddings: [qEmbed2], nResults: 2, include: ['documents', 'metadatas', 'distances'] as any });
  printQueryResults('Text Query 2 — expected: text_3', q2, r2);
}

async function imageExample() {
  console.log('\n═══════════════════ IMAGE TEST ═══════════════════');

  const COLLECTION = 'test_images';
  const filePaths = ['1.png', '2.png', '3.png', '4.png', '5.png'].map(
    (f) => `./public/images/${f}`
  );
  const ids       = filePaths.map((_, i) => `image_${i + 1}`);
  const metadatas = [
    { subject: 'rabbit', setting: 'nature'  },
    { subject: 'dog',    setting: 'nature'  },
    { subject: 'cat',    setting: 'nature'  },
    { subject: 'ship',   setting: 'sea'     },
    { subject: 'car',    setting: 'street'  },
  ];

  const buffers    = await Promise.all(filePaths.map((p) => fs.readFile(p)));
  const embeddings = await embedImages(buffers, 'image/png');
  printEmbeddingStats('Image embeddings', embeddings);

  await storeImages(embeddings, { collection: COLLECTION, ids, metadatas });

  const collection = await chroma.getOrCreateCollection({ name: COLLECTION });

  // Query 1: text → image (cross-modal retrieval)
  const q1 = 'A dog playing in the park';
  const [qEmbed1] = await embedTexts([q1]);
  const r1 = await collection.query({ queryEmbeddings: [qEmbed1], nResults: 2, include: ['documents', 'metadatas', 'distances'] as any });
  printQueryResults('Image Query 1 (text→image) — expected: image_2', q1, r1);

  // Query 2: image → image (nearest neighbour)
  const queryBuffer = await fs.readFile('./public/images/2.png'); // dog image
  const [qEmbed2]   = await embedImages([queryBuffer], 'image/png');
  const r2          = await collection.query({ queryEmbeddings: [qEmbed2], nResults: 2, include: ['documents', 'metadatas', 'distances'] as any });
  printQueryResults('Image Query 2 (image→image) — expected: image_2 (itself)', 'image_2.png as query image', r2);
}

async function videoExample() {
  console.log('\n═══════════════════ VIDEO TEST ═══════════════════');

  const COLLECTION = 'test_videos';
  const filePaths  = ['1.mp4', '2.mp4', '3.mp4'].map((f) => `./public/videos/${f}`);
  const ids        = filePaths.map((_, i) => `video_${i + 1}`);
  const metadatas  = [
    { subject: 'airplane',            setting: 'sky'       },
    { subject: 'woman and dog',       setting: 'outdoors'  },
    { subject: 'teacher and students', setting: 'classroom' },
  ];

  const buffers    = await Promise.all(filePaths.map((p) => fs.readFile(p)));
  const embeddings = await embedVideos(buffers, 'video/mp4');
  printEmbeddingStats('Video embeddings', embeddings);

  await storeVideos(embeddings, { collection: COLLECTION, ids, metadatas });

  const collection = await chroma.getOrCreateCollection({ name: COLLECTION });

  const q1 = 'A dog playing in the park';
  const [qEmbed1] = await embedTexts([q1]);
  const r1 = await collection.query({ queryEmbeddings: [qEmbed1], nResults: 2, include: ['documents', 'metadatas', 'distances'] as any });
  printQueryResults('Video Query 1 — expected: video_2', q1, r1);

  const q2 = 'An educational environment';
  const [qEmbed2] = await embedTexts([q2]);
  const r2 = await collection.query({ queryEmbeddings: [qEmbed2], nResults: 2, include: ['documents', 'metadatas', 'distances'] as any });
  printQueryResults('Video Query 2 — expected: video_3', q2, r2);
}

async function audioExample() {
  console.log('\n═══════════════════ AUDIO TEST ═══════════════════');

  const COLLECTION = 'test_audios';
  const filePaths  = ['1.wav'].map((f) => `./public/audios/${f}`);
  const ids        = ['audio_1'];
  const metadatas  = [{ topic: 'food', duration_sec: 17 }];

  const buffers    = await Promise.all(filePaths.map((p) => fs.readFile(p)));
  const embeddings = await embedAudios(buffers, 'audio/wav');
  printEmbeddingStats('Audio embeddings', embeddings);

  await storeAudios(embeddings, { collection: COLLECTION, ids, metadatas });

  const collection = await chroma.getOrCreateCollection({ name: COLLECTION });

  const q1 = 'A person talking about food';
  const [qEmbed1] = await embedTexts([q1]);
  const r1 = await collection.query({ queryEmbeddings: [qEmbed1], nResults: 1, include: ['documents', 'metadatas', 'distances'] as any });
  printQueryResults('Audio Query 1 — expected: audio_1', q1, r1);
}

async function fileExample() {
  console.log('\n═══════════════════ FILE TEST ═══════════════════');

  const COLLECTION = 'test_files';
  const filePaths  = ['1.txt', '2.txt'].map((f) => `./public/files/${f}`);
  const ids        = ['file_1', 'file_2'];
  const metadatas  = [
    { topic: 'time management',          type: 'article' },
    { topic: 'technology and education', type: 'article' },
  ];

  const buffers    = await Promise.all(filePaths.map((p) => fs.readFile(p)));
  const embeddings = await embedFiles(buffers, 'text/plain');
  printEmbeddingStats('File embeddings', embeddings);

  await storeFiles(embeddings, { collection: COLLECTION, ids, metadatas });

  const collection = await chroma.getOrCreateCollection({ name: COLLECTION });

  const q1 = 'Methods for managing time effectively';
  const [qEmbed1] = await embedTexts([q1]);
  const r1 = await collection.query({ queryEmbeddings: [qEmbed1], nResults: 2, include: ['documents', 'metadatas', 'distances'] as any });
  printQueryResults('File Query 1 — expected: file_1', q1, r1);

  const q2 = 'Impact of digital tools on students';
  const [qEmbed2] = await embedTexts([q2]);
  const r2 = await collection.query({ queryEmbeddings: [qEmbed2], nResults: 2, include: ['documents', 'metadatas', 'distances'] as any });
  printQueryResults('File Query 2 — expected: file_2', q2, r2);
}

// ─── Main Runner ─────────────────────────────────────────────────────────────

async function runAll() {
  const examples: [string, () => Promise<void>][] = [
    ['Text',  textExample],
    ['Image', imageExample],
    ['Video', videoExample],
    ['Audio', audioExample],
    ['File',  fileExample],
  ];

  let passed = 0;
  let failed = 0;

  for (const [name, fn] of examples) {
    try {
      await fn();
      passed++;
    } catch (err) {
      console.error(`\n❌ ${name} test FAILED:`, err instanceof Error ? err.message : err);
      failed++;
    }
  }

  console.log(`\n${'═'.repeat(60)}`);
  console.log(`🏁 Results: ${passed} passed / ${failed} failed`);
  console.log(`${'═'.repeat(60)}\n`);
}

runAll();