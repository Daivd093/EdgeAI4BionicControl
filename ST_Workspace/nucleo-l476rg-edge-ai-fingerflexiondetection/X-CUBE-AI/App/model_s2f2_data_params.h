/**
  ******************************************************************************
  * @file    model_s2f2_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-07-01T08:23:35-0400
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

#ifndef MODEL_S2F2_DATA_PARAMS_H
#define MODEL_S2F2_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_MODEL_S2F2_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_model_s2f2_data_weights_params[1]))
*/

#define AI_MODEL_S2F2_DATA_CONFIG               (NULL)


#define AI_MODEL_S2F2_DATA_ACTIVATIONS_SIZES \
  { 4612, }
#define AI_MODEL_S2F2_DATA_ACTIVATIONS_SIZE     (4612)
#define AI_MODEL_S2F2_DATA_ACTIVATIONS_COUNT    (1)
#define AI_MODEL_S2F2_DATA_ACTIVATION_1_SIZE    (4612)



#define AI_MODEL_S2F2_DATA_WEIGHTS_SIZES \
  { 156612, }
#define AI_MODEL_S2F2_DATA_WEIGHTS_SIZE         (156612)
#define AI_MODEL_S2F2_DATA_WEIGHTS_COUNT        (1)
#define AI_MODEL_S2F2_DATA_WEIGHT_1_SIZE        (156612)



#define AI_MODEL_S2F2_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_model_s2f2_activations_table[1])

extern ai_handle g_model_s2f2_activations_table[1 + 2];



#define AI_MODEL_S2F2_DATA_WEIGHTS_TABLE_GET() \
  (&g_model_s2f2_weights_table[1])

extern ai_handle g_model_s2f2_weights_table[1 + 2];


#endif    /* MODEL_S2F2_DATA_PARAMS_H */
