/**
  ******************************************************************************
  * @file    sine_model_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-06-18T02:44:30-0400
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "sine_model_data_params.h"


/**  Activations Section  ****************************************************/
ai_handle g_sine_model_activations_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(NULL),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};




/**  Weights Section  ********************************************************/
AI_ALIGNED(32)
const ai_u64 s_sine_model_weights_array_u64[53] = {
  0xb996229a3800c665U, 0xe27fcab9333f7002U, 0xfffff273U, 0xffffe8dfffffff8aU,
  0x153200000000U, 0x0U, 0xffffe96e00001588U, 0xffffece4000010b9U,
  0x0U, 0xffffff8eU, 0x1bc23bff0def19d1U, 0xcc3b1cd322e305beU,
  0xccfb0de762de2c1fU, 0x29f3e3f7e7d0de0dU, 0xd1d226e9121ff42aU, 0x41c4ccfe04deba23U,
  0x1eb94c0502ec10edU, 0x23e72146c6f3a944U, 0x7e20c120b09f9ccU, 0xeb2135e53d123fccU,
  0x39ae0a3a0181f614U, 0xe2dad3ea350d34dcU, 0xe2cc312af23c3dd2U, 0x1cdbfaeb2cc80829U,
  0x360fda043018e5d6U, 0x14cde6bf1cf604eaU, 0xc70ae2caead323f5U, 0xf9dff2fbf53cce23U,
  0x39ee1c1926e603dcU, 0x9d0d92151e653e6U, 0x280f3301c4fa3012U, 0xe2c60f331906d3c5U,
  0xccca00f10ffb34daU, 0xfc44313239ddee02U, 0xffe4eed708eee4b5U, 0xf2272afc9613281dU,
  0x40cfd9205ec6ca3fU, 0xd650c20bfee113ebU, 0xd9003725e8e41300U, 0xcf2ac634e135ac35U,
  0xd82eb53a3f2a0b1cU, 0x70919d460de45a2U, 0xfffff2ad00000000U, 0x8c4000006d8U,
  0xfffff1a5U, 0xffffffe400000000U, 0xffffffe4fffff40cU, 0xfffff46600000000U,
  0xfffff3bf00000928U, 0xfffff12800000966U, 0xeb25f76abfd05af3U, 0x7f25db15ddfdebe5U,
  0x629U,
};


ai_handle g_sine_model_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_sine_model_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

