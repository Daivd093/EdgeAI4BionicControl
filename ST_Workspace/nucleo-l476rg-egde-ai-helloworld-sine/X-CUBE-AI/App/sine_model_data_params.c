/**
  ******************************************************************************
  * @file    sine_model_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-05-13T09:00:38-0400
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
const ai_u64 s_sine_model_weights_array_u64[161] = {
  0xbea5e96a3f108edfU, 0x3ea057d93b20ac12U, 0x3e41649fbf125cc9U, 0xbecb17d8bf173243U,
  0x3f20a0833c2764e5U, 0x3e931a5e3eb494d9U, 0xbe9a1b36bec9f29cU, 0xbe2e1e6e3f358756U,
  0xbef36623U, 0xbf4fb8f6bc84aaddU, 0x3f3e560700000000U, 0x0U,
  0xbf4aad513f415aaeU, 0xbf2b99203f16286dU, 0x0U, 0xbc80861aU,
  0x3e2bb8f6bea03f79U, 0x3db09ec4bde58878U, 0xbe675e1a3db19644U, 0xbe2a68b4bea31b0cU,
  0x3dd6a6acbdffd878U, 0x3c61d540be079ca8U, 0x3ed345bdbeae5181U, 0x3ecedd993df3b4ccU,
  0x3ec64cbfbbdd447fU, 0x3e332345bed21610U, 0xbe43ed40bd7f0a18U, 0x3e89a543bdb00d0cU,
  0x3effffb43d0d7e53U, 0x3e476d03beeea8e2U, 0x3eb31975be382718U, 0xbe0acfea3e5ec89cU,
  0x3d12d438bedeff19U, 0x3e6504febe434e68U, 0xbda72a703e8eae9dU, 0x3def93cc3e5350c2U,
  0xbf12c5c33ee66392U, 0xbec4b0e8bdab2c99U, 0xbd81ba643e0501d2U, 0x3c03b260bf565a5dU,
  0x3e3de4b6be973b16U, 0xbeae7d213ec6500fU, 0x3e7e7cf4be1b4892U, 0xbe9fe2b7be9c62e8U,
  0x3e5f2f953eebcdf3U, 0x3e6d8a86be25d315U, 0x3d897f3c3ec27ef1U, 0x3ec0d149bf0b3d31U,
  0x3e9560793e531789U, 0x3f2587eabe632f55U, 0xbeeb45763e6d47deU, 0x3ce651a0be673d23U,
  0xbd3cf60abeae3777U, 0x3d999a1c3d76142bU, 0x3eaf5067be71bafbU, 0x3eb30d693daffeccU,
  0x3db2cc34be29fd20U, 0xbeb016a3bd012700U, 0xbeae3828bc854130U, 0x3edb21b7bec8d9beU,
  0x3da39ebc3dec799cU, 0x3d453b00be4b9daaU, 0xbe987f54be129a9eU, 0xbe4cd4f2be7d6afeU,
  0x3ece3241be9ce73cU, 0xbdbeba143ec9ad91U, 0x3ce650c0be174daaU, 0x3e39bbdabd8b9c70U,
  0x3cbace90be70fdddU, 0x3e815451be2c42daU, 0xbe96d854bec69f04U, 0x3e2915ca3d189728U,
  0x3ea5de933e8e9b27U, 0xbe47edf0beafe0b7U, 0xbe319ea4bedbf6ccU, 0x3e099cd6beac892aU,
  0x3e3e60da3e2b3f4fU, 0x3ec0b9afbdf81d1cU, 0x3dc894443eabac79U, 0xbe4b3d28bec4306dU,
  0x3d4df3a03e8c0802U, 0x3e93df54bebce480U, 0x3e6fa8cabd9531b8U, 0xbe12555cbe9930baU,
  0x3f0b8eb1be2dab4aU, 0x3f08d515be2e2edcU, 0x3eaf0ab9be812c62U, 0x3dc9ffe4bd0c4db8U,
  0xbd202308be0e5480U, 0x3e3ef136be7c218cU, 0xbe49a5c6beb66d07U, 0xbec1fd9a3d8400f4U,
  0xbe848c6e3e5d35e4U, 0x3d759310bea0fec9U, 0x3919b800bdc75410U, 0xbeaf6ba4beb618e7U,
  0xbe344384be8c1ab0U, 0x3ea31cd33e2532b2U, 0xbea9bcc23e6aff5aU, 0xbd9b36d03eca74b3U,
  0x3ea05a933df2ffbcU, 0xbecadb7fbd19bbe0U, 0xbdf925083c7574a0U, 0x3ec09c51be6efa09U,
  0xbe812a723cbfb0f2U, 0x3eb5417e3dcb0ef1U, 0xbdb7510cbd098c10U, 0xbd34c0b8be60f533U,
  0x3eabc4943bea341fU, 0x3e87f7883dc52e93U, 0x3ea58bc33ea72913U, 0xbcd583703ee695caU,
  0xbe3f0d84befc2969U, 0x3d57176abdf024cfU, 0xbded2765be8992d8U, 0xbbf86680be3c3e10U,
  0x3e8635293e46bdfaU, 0xbf338c773e02349fU, 0x3e8c26a5bcc94db0U, 0xbdbf331c3e82dc69U,
  0xbeb4c84d3ed50c35U, 0x3f1e082fbec3945fU, 0xbe82b8753e57b1ceU, 0x3ed6e70dbea67fe4U,
  0x3e01e105be10d232U, 0xbc30865dbe53a584U, 0xbed0fc433d8fb924U, 0xbe8c81943f06b294U,
  0x3e026a7e3931420cU, 0xbe247438be3e4c38U, 0x3eb93e7c3e795f4eU, 0xbe83acc0ba706800U,
  0xbf0d5d0e3eb1cbaeU, 0xbe5224f73eb48b4dU, 0xbec505ce3eae127dU, 0xbea593963e8e7bdaU,
  0x3d932fcc3e3a197dU, 0x3ed5ebf63e8c73ccU, 0xbefe76d23ec30db7U, 0xbe86ed273e99c199U,
  0x3eea81b3bf1eec02U, 0x3f213687be678cecU, 0x3e2adcf6be932e92U, 0x3d4752a83d795542U,
  0xbec7a7a000000000U, 0x3e8358503e4d0f13U, 0xbed71b26U, 0xbb4f37f400000000U,
  0xbb4f37debeb326e9U, 0xbeadd8b200000000U, 0xbeb79b1c3e89311cU, 0xbede72123e8cdb0dU,
  0x3f8f1b6bbe252b4aU, 0xbf4d4eebbf19d3c8U, 0xbdda4f403fa86f85U, 0xbe88291d3eeac104U,
  0xbe85df44beaba5e8U, 0xbedc0cd9bd207390U, 0xbeeb040e3e876a63U, 0x3fc96bde3ee9990eU,
  0x3e8c9002U,
};


ai_handle g_sine_model_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_sine_model_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

