import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';
import {
  AIAnalysisResult,
  DefectObservation,
  DefectType,
  DefectSeverity,
  ShapeCategory,
  SizeCategory,
  SurfaceCondition,
  VegetableType,
  VegetableDetectionTarget,
} from '../../models/types';
import { IAIService, ImageInput, ImageValidationResult } from './AIService';

const SUPPORTED_VEGETABLES: VegetableType[] = [
  'potato',
  'tomato',
  'onion',
  'carrot',
  'bell_pepper',
  'cucumber',
];

export class GeminiAIService implements IAIService {
  private ai: GoogleGenAI;

  constructor() {
    this.ai = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY || '',
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }

  /**
   * Fast pre-check of image quality heuristic
   */
  public async validateImageQuality(image: ImageInput): Promise<ImageValidationResult> {
    const issues: string[] = [];

    if (!image.base64Data || image.base64Data.length < 100) {
      issues.push('Image data payload is missing or too small.');
      return { isValid: false, issues };
    }

    const byteSize = (image.base64Data.length * 3) / 4;
    if (byteSize < 100) {
      issues.push('Image resolution is insufficient for defect detection.');
    }

    return {
      isValid: issues.length === 0,
      issues,
    };
  }

  /**
   * Normalize MIME type to standard IANA values accepted by Gemini Vision
   */
  private normalizeMimeType(rawMime: string): string {
    const lower = (rawMime || '').toLowerCase().trim();
    if (lower === 'image/jpg' || lower === 'image/pjpeg') return 'image/jpeg';
    if (lower === 'image/png') return 'image/png';
    if (lower === 'image/webp') return 'image/webp';
    if (lower === 'image/heic') return 'image/heic';
    if (lower === 'image/heif') return 'image/heif';
    return 'image/jpeg';
  }

  /**
   * Clean base64 string
   */
  private cleanBase64String(raw: string): string {
    return raw
      .replace(/^data:image\/[a-z0-9.+_-]+;base64,/i, '')
      .replace(/[\r\n\s]/g, '');
  }

  public async analyzePotato(image: ImageInput): Promise<AIAnalysisResult> {
    return this.analyzeProduce(image, 'potato');
  }

  /**
   * Multi-Commodity AI Vision inspection with auto-detection and model failover
   */
  public async analyzeProduce(
    image: ImageInput,
    targetVeggie: VegetableDetectionTarget = 'auto'
  ): Promise<AIAnalysisResult> {
    const validation = await this.validateImageQuality(image);
    if (!validation.isValid) {
      return {
        vegetableType: targetVeggie === 'auto' ? 'unknown' : targetVeggie,
        detected: false,
        detectionConfidence: 0,
        detectionMessage: 'Image quality is insufficient for reliable analysis.',
        defects: [],
        shape: { category: 'unknown', confidence: 0 },
        size: {
          category: 'unknown',
          confidence: 0,
          qualitativeNotes: 'Cannot evaluate size from unusable image.',
        },
        surface: { condition: 'unknown', confidence: 0 },
        overallConfidence: 0,
        imageQualityOk: false,
        imageQualityIssues: validation.issues,
        rawObservationsSummary: 'Rejected prior to inference due to invalid image payload.',
      };
    }

    const cleanBase64 = this.cleanBase64String(image.base64Data);
    const mimeType = this.normalizeMimeType(image.mimeType);

    const veggieDescriptions: Record<VegetableType, { name: string; scientific: string; defectsHint: string }> = {
      potato: {
        name: 'potato',
        scientific: 'Solanum tuberosum tuber',
        defectsHint: 'greening (chlorophyll/solanine), bruising, cuts, cracks, wet soft rot, late blight, scab, sprouting',
      },
      tomato: {
        name: 'tomato',
        scientific: 'Solanum lycopersicum fruit',
        defectsHint: 'blossom-end rot, radial/concentric cracks, catfacing, sunscald, bruising, soft decay',
      },
      onion: {
        name: 'onion',
        scientific: 'Allium cepa bulb',
        defectsHint: 'neck rot, black mold spores, green sprouting from neck, translucent scales, basal plate rot',
      },
      carrot: {
        name: 'carrot',
        scientific: 'Daucus carota taproot',
        defectsHint: 'forking/branching, crown greening/sunburn, cavity spots, root growth cracks, soft rot',
      },
      bell_pepper: {
        name: 'bell pepper',
        scientific: 'Capsicum annuum fruit',
        defectsHint: 'blossom-end rot, bleached sunscald, circular anthracnose spots, chilling injury pitting, soft decay',
      },
      cucumber: {
        name: 'cucumber',
        scientific: 'Cucumis sativus fruit',
        defectsHint: 'crooking/curvature > 15 deg, yellowing (over-maturity), belly rot, spongy softness, blossom shriveling',
      },
    };

    const isAutoMode = targetVeggie === 'auto' || !(targetVeggie in veggieDescriptions);
    const targetInfo = !isAutoMode ? veggieDescriptions[targetVeggie as VegetableType] : null;

    const supportedCatalogText = SUPPORTED_VEGETABLES.map((id) => {
      const info = veggieDescriptions[id];
      return `- "${id}" (${info.name}): ${info.defectsHint}`;
    }).join('\n');

    const detectionClause = isAutoMode
      ? `Mode: AUTO-DETECT. Supported IDs:
${supportedCatalogText}
1. DETECTION: Identify if primary subject is one of the 6 IDs above. If yes -> detected=true, vegetableType=exact ID, detectionConfidence=0.85-0.98. If not (other produce/object/background) -> detected=false, vegetableType="unknown", detectionConfidence<=0.4, explain briefly in detectionMessage.`
      : `Target: ${targetInfo!.name.toUpperCase()} (${targetInfo!.scientific}).
1. DETECTION: If subject is a ${targetInfo!.name} -> detected=true, vegetableType="${targetVeggie}", detectionConfidence=0.85-0.98. Otherwise -> detected=false, vegetableType="unknown", detectionConfidence<=0.4, explain briefly in detectionMessage.`;

    const defectsClause = isAutoMode
      ? `4. DEFECTS: Check visible external defects for the detected vegetableType. Keep visibleEvidence to 1 short sentence.`
      : `4. DEFECTS: Check ${targetInfo!.defectsHint}. Keep visibleEvidence to 1 short sentence.`;

    const systemInstruction = `You are AgriGrade Vision, a produce surface inspector.
${detectionClause}
2. OBSERVATION ONLY: Never assign Grade A, B, or Reject.
3. VISIBLE EVIDENCE ONLY: Record visible external features only (no internal/chemical claims).
${defectsClause}
5. SHAPE: 'regular' | 'slightly_irregular' | 'irregular' | 'unknown'.
6. SIZE: 'small' | 'medium' | 'large' | 'unknown' (visual qualitative estimate only).
7. SURFACE: 'clean' | 'mostly_clean' | 'soil_covered' | 'blemished' | 'unknown'.
8. CONFIDENCE: Assign realistic 0.0-1.0 confidence scores. Keep all text fields brief (max 15 words).`;

    const promptText = isAutoMode
      ? `Identify the vegetable ("potato"|"tomato"|"onion"|"carrot"|"bell_pepper"|"cucumber" or "unknown"), inspect visible surface defects, shape, size, and surface condition. Return concise JSON.`
      : `Verify if this is a ${targetInfo!.name}, inspect visible surface defects, shape, size, and surface condition. Return concise JSON.`;

    const responseSchema = {
      type: Type.OBJECT,
      properties: {
        vegetableType: {
          type: Type.STRING,
          description: isAutoMode
            ? 'Must be one of: "potato", "tomato", "onion", "carrot", "bell_pepper", "cucumber", or "unknown" if not detected.'
            : `Must be "${targetVeggie}" if detected, or "unknown" if not detected.`,
        },
        detected: {
          type: Type.BOOLEAN,
          description: isAutoMode
            ? 'True if one of the 6 supported vegetables is clearly visible'
            : `True if a ${targetInfo!.name} is clearly visible`,
        },
        detectionConfidence: { type: Type.NUMBER, description: 'Confidence between 0.0 and 1.0' },
        detectionMessage: {
          type: Type.STRING,
          description: 'Short explanation of what vegetable was identified, or why a supported vegetable was not detected',
        },
        imageQualityOk: { type: Type.BOOLEAN, description: 'True if lighting, focus, and framing are sufficient' },
        imageQualityIssues: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description: 'List of photographic quality problems if any (e.g. severe blur, dark, glare)',
        },
        defects: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              type: {
                type: Type.STRING,
                description: 'discoloration | greening | bruising | cuts | cracks | rot | blight | scab | insect_damage | sprouting | surface_blemish | unknown',
              },
              label: { type: Type.STRING, description: 'Human readable defect name' },
              severity: { type: Type.STRING, description: 'minor | moderate | severe' },
              confidence: { type: Type.NUMBER, description: '0.0 to 1.0' },
              locationDescription: { type: Type.STRING, description: 'Location on the produce item' },
              visibleEvidence: { type: Type.STRING, description: 'Visible physical description' },
            },
            required: ['type', 'label', 'severity', 'confidence', 'visibleEvidence'],
          },
        },
        shape: {
          type: Type.OBJECT,
          properties: {
            category: { type: Type.STRING, description: 'regular | slightly_irregular | irregular | unknown' },
            confidence: { type: Type.NUMBER },
            description: { type: Type.STRING },
          },
          required: ['category', 'confidence'],
        },
        size: {
          type: Type.OBJECT,
          properties: {
            category: { type: Type.STRING, description: 'small | medium | large | unknown' },
            confidence: { type: Type.NUMBER },
            qualitativeNotes: { type: Type.STRING, description: 'Explicit visual disclaimer' },
          },
          required: ['category', 'confidence', 'qualitativeNotes'],
        },
        surface: {
          type: Type.OBJECT,
          properties: {
            condition: { type: Type.STRING, description: 'clean | mostly_clean | soil_covered | blemished | unknown' },
            confidence: { type: Type.NUMBER },
          },
          required: ['condition', 'confidence'],
        },
        overallConfidence: { type: Type.NUMBER, description: 'Overall AI observation confidence 0.0 to 1.0' },
        rawObservationsSummary: { type: Type.STRING, description: 'Concise scientific summary of observations' },
      },
      required: [
        'vegetableType',
        'detected',
        'detectionConfidence',
        'imageQualityOk',
        'imageQualityIssues',
        'defects',
        'shape',
        'size',
        'surface',
        'overallConfidence',
        'rawObservationsSummary',
      ],
    };

    if (!process.env.GEMINI_API_KEY) {
      const keyErr: any = new Error(
        'GEMINI_API_KEY is not configured. Please verify your API key in Settings > Secrets.'
      );
      keyErr.status = 401;
      throw keyErr;
    }

    const modelsToTry: Array<{ name: string; thinkingLevel?: ThinkingLevel }> = [
      { name: 'gemini-4-hydrogen', thinkingLevel: ThinkingLevel.LOW },
      { name: 'gemini-3.1-flash-lite' },
    ];
    let lastError: any = null;

    for (const modelSpec of modelsToTry) {
      const maxAttempts = 2;
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        const startTime = Date.now();
        try {
          console.log(
            `[GeminiAIService] Running ${targetVeggie} inspection with ${modelSpec.name} (attempt ${attempt}/${maxAttempts})`
          );
          const config: any = {
            systemInstruction,
            responseMimeType: 'application/json',
            responseSchema,
          };
          if (modelSpec.thinkingLevel) {
            config.thinkingConfig = { thinkingLevel: modelSpec.thinkingLevel };
          }

          const response = await this.ai.models.generateContent({
            model: modelSpec.name,
            contents: [
              {
                role: 'user',
                parts: [
                  {
                    inlineData: {
                      mimeType,
                      data: cleanBase64,
                    },
                  },
                  { text: promptText },
                ],
              },
            ],
            config,
          });

          const responseText = response.text?.trim() || '{}';
          const parsed = JSON.parse(responseText);

          console.log(
            `[GeminiAIService] Success with ${modelSpec.name} in ${Date.now() - startTime}ms! Veggie:`,
            parsed.vegetableType,
            'Detected:',
            parsed.detected,
            'Confidence:',
            parsed.overallConfidence
          );
          return this.sanitizeAIOutput(parsed, targetVeggie);
        } catch (err: any) {
          lastError = err;
          const status = err?.status || err?.code;
          const isTransient = status === 503 || status === 429 || status === 500 || status === 502;
          console.log(
            `[GeminiAIService] ${modelSpec.name} attempt ${attempt} did not complete in ${Date.now() - startTime}ms (status=${status || 'unknown'}).`
          );

          // If non-transient (e.g. 400/403/404), move immediately to fallback or fail
          if (!isTransient) break;

          if (attempt < maxAttempts) {
            await new Promise((r) => setTimeout(r, 400 * attempt));
          }
        }
      }
    }

    const lastStatus = lastError?.status || 503;
    const rawMsg = lastError instanceof Error ? lastError.message : 'Service temporarily unavailable';
    let friendlyMsg = `AI Vision service temporarily unavailable (${rawMsg}). Please tap "Retry Analysis" in a moment.`;

    if (lastStatus === 403 || lastStatus === 400) {
      friendlyMsg = `Gemini API authentication or request error (${rawMsg}). Please verify your API key in Settings > Secrets.`;
    } else if (lastStatus === 429) {
      friendlyMsg = `Gemini API rate limit reached (${rawMsg}). Please wait a moment and tap "Retry Analysis", or check your quota in Settings > Secrets.`;
    } else if (lastStatus === 503) {
      friendlyMsg = 'AI Vision service is experiencing temporary high demand (503). Please tap "Retry Analysis" in a few seconds.';
    }

    console.error(`[GeminiAIService] All models failed (status=${lastStatus}):`, rawMsg);
    const outErr: any = new Error(friendlyMsg);
    outErr.status = lastStatus;
    throw outErr;
  }

  private sanitizeAIOutput(raw: any, targetVeggie: VegetableDetectionTarget): AIAnalysisResult {
    const rawTypeNormalized = String(raw.vegetableType || '')
      .toLowerCase()
      .trim()
      .replace(/[\s-]+/g, '_');

    const matchedSupported = SUPPORTED_VEGETABLES.find((v) => v === rawTypeNormalized);
    let isDetected = Boolean(raw.detected);
    const detectionConf = Math.min(Math.max(Number(raw.detectionConfidence) || 0, 0), 1);

    let resolvedVeggie: VegetableType | 'unknown';
    if (targetVeggie === 'auto') {
      if (isDetected && matchedSupported) {
        resolvedVeggie = matchedSupported;
      } else {
        resolvedVeggie = matchedSupported || 'unknown';
        if (isDetected && !matchedSupported) {
          isDetected = false;
        }
      }
    } else {
      resolvedVeggie = isDetected ? targetVeggie : matchedSupported || 'unknown';
    }

    const validSeverities: DefectSeverity[] = ['minor', 'moderate', 'severe'];
    const validShapes: ShapeCategory[] = ['regular', 'slightly_irregular', 'irregular', 'unknown'];
    const validSizes: SizeCategory[] = ['small', 'medium', 'large', 'unknown'];
    const validSurfaces: SurfaceCondition[] = ['clean', 'mostly_clean', 'soil_covered', 'blemished', 'unknown'];

    const defects: DefectObservation[] = Array.isArray(raw.defects)
      ? raw.defects.map((d: any, idx: number) => ({
          id: `defect_${Date.now()}_${idx}`,
          type: (d.type as DefectType) || 'unknown',
          label: String(d.label || d.type || 'Defect'),
          severity: validSeverities.includes(d.severity) ? d.severity : 'minor',
          confidence: Math.min(Math.max(Number(d.confidence) || 0.85, 0), 1),
          locationDescription: d.locationDescription ? String(d.locationDescription) : undefined,
          visibleEvidence: String(
            d.visibleEvidence || `Visible surface ${d.label || d.type || 'irregularity'} detected.`
          ),
        }))
      : [];

    const shapeCategory = validShapes.includes(raw.shape?.category)
      ? raw.shape.category
      : 'unknown';
    const sizeCategory = validSizes.includes(raw.size?.category)
      ? raw.size.category
      : 'unknown';
    const surfaceCondition = validSurfaces.includes(raw.surface?.condition)
      ? raw.surface.condition
      : 'unknown';

    const fallbackNotDetectedMsg =
      targetVeggie === 'auto'
        ? 'No supported vegetable (tomato, bell pepper, carrot, onion, cucumber, or potato) detected in the frame.'
        : `${targetVeggie.replace('_', ' ').toUpperCase()} not detected in input frame.`;

    return {
      vegetableType: resolvedVeggie,
      detected: isDetected,
      detectionConfidence: detectionConf,
      detectionMessage: raw.detectionMessage || (!isDetected ? fallbackNotDetectedMsg : undefined),
      imageQualityOk: Boolean(raw.imageQualityOk ?? true),
      imageQualityIssues: Array.isArray(raw.imageQualityIssues) ? raw.imageQualityIssues.map(String) : [],
      defects,
      shape: {
        category: shapeCategory,
        confidence: Math.min(Math.max(Number(raw.shape?.confidence) || 0.88, 0), 1),
        description: raw.shape?.description ? String(raw.shape.description) : undefined,
      },
      size: {
        category: sizeCategory,
        confidence: Math.min(Math.max(Number(raw.size?.confidence) || 0.85, 0), 1),
        qualitativeNotes: String(
          raw.size?.qualitativeNotes ||
            'Visual qualitative estimate based on frame framing; no metric millimeter claim.'
        ),
      },
      surface: {
        condition: surfaceCondition,
        confidence: Math.min(Math.max(Number(raw.surface?.confidence) || 0.88, 0), 1),
      },
      overallConfidence: Math.min(Math.max(Number(raw.overallConfidence) || 0.85, 0), 1),
      rawObservationsSummary: String(
        raw.rawObservationsSummary || 'Visual inspection recorded without anomalies.'
      ),
    };
  }
}
