
static float bf16_to_float(uint16_t bf16) {
  uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  float fp32;
  std::memcpy(&fp32, &bits, sizeof(fp32));
  return fp32;
}

static uint16_t float_to_bf16(float fp32) {
  uint32_t bits;
  std::memcpy(&bits, &fp32, sizeof(fp32));
  return static_cast<uint16_t>(bits >> 16);
}

/************************************************
 * Copyright (c) 2015 Princeton Vision Group
 * Licensed under the MIT license.
 * Codes below copied from
 * https://github.com/PrincetonVision/marvin/tree/master/tools/tensorIO_matlab
 *************************************************/
static uint16_t float_to_fp16(float fp32) {
  uint16_t fp16;

  unsigned x;
  unsigned u, remainder, shift, lsb, lsb_s1, lsb_m1;
  unsigned sign, exponent, mantissa;

  std::memcpy(&x, &fp32, sizeof(fp32));
  u = (x & 0x7fffffff);

  // Get rid of +NaN/-NaN case first.
  if (u > 0x7f800000) {
    fp16 = 0x7fffU;
    return fp16;
  }

  sign = ((x >> 16) & 0x8000);

  // Get rid of +Inf/-Inf, +0/-0.
  if (u > 0x477fefff) {
    fp16 = sign | 0x7c00U;
    return fp16;
  }
  if (u < 0x33000001) {
    fp16 = (sign | 0x0000);
    return fp16;
  }

  exponent = ((u >> 23) & 0xff);
  mantissa = (u & 0x7fffff);

  if (exponent > 0x70) {
    shift = 13;
    exponent -= 0x70;
  } else {
    shift = 0x7e - exponent;
    exponent = 0;
    mantissa |= 0x800000;
  }
  lsb = (1 << shift);
  lsb_s1 = (lsb >> 1);
  lsb_m1 = (lsb - 1);

  // Round to nearest even.
  remainder = (mantissa & lsb_m1);
  mantissa >>= shift;
  if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
    ++mantissa;
    if (!(mantissa & 0x3ff)) {
      ++exponent;
      mantissa = 0;
    }
  }

  fp16 = (sign | (exponent << 10) | mantissa);

  return fp16;
}

static float fp16_to_float(uint16_t fp16) {
  unsigned sign = ((fp16 >> 15) & 1);
  unsigned exponent = ((fp16 >> 10) & 0x1f);
  unsigned mantissa = ((fp16 & 0x3ff) << 13);
  int temp;
  float fp32;
  if (exponent == 0x1f) { /* NaN or Inf */
    mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
    exponent = 0xff;
  } else if (!exponent) { /* Denorm or Zero */
    if (mantissa) {
      unsigned int msb;
      exponent = 0x71;
      do {
        msb = (mantissa & 0x400000);
        mantissa <<= 1; /* normalize */
        --exponent;
      } while (!msb);
      mantissa &= 0x7fffff; /* 1.mantissa is implicit */
    }
  } else {
    exponent += 0x70;
  }
  temp = ((sign << 31) | (exponent << 23) | mantissa);
  std::memcpy(&fp32, &temp, sizeof(temp));
  return fp32;
}
