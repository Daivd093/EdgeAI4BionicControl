/**
  ******************************************************************************
  * @file    sine_model_data_params.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-05-04T20:06:56-0400
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
  0xbcd6e020be8442c7U, 0x3e3129583f0b8bcdU, 0x3e90f2753f2f2b17U, 0xbf0eab3fbf0cdebcU,
  0x3e5ec9473f0763abU, 0xbe993d993eb493baU, 0xbebde80cbf17a268U, 0x3f445de13e84b5b6U,
  0x0U, 0x3f8fcf91bf012bacU, 0x3e38ec623e3cf0deU, 0x0U,
  0xbe73c367bef996f5U, 0xbe67934eU, 0x0U, 0xbe09bf41bf212a7fU,
  0xbe05ca923e8bfbe3U, 0xbd6b85963ec0f91cU, 0x3ec643f73e67956dU, 0xbe9f06f23e0b540aU,
  0xbe93cb6cbe420014U, 0x3d1c8a783da1cd0eU, 0xbe3dbcbdbecfcbdeU, 0x3ed176503dccd3cdU,
  0x3e05c44a3debccccU, 0xbf1e626b3e1c9851U, 0xbe2555503e99a18bU, 0x3eb4ef27be2cfd4aU,
  0xbe0688ae3eb5f64dU, 0xbd23e4f8be3cc40dU, 0xbeaa58773cebdfa0U, 0x3c24952d3e0a2f63U,
  0x3e4ea986be7ebce0U, 0x3e311e903e9a55afU, 0x3e5b3b4bbda08606U, 0xbd7666b03e1a239aU,
  0xbe9266ebbd9a9825U, 0xbea653b2beba5308U, 0x3e273c7a3ebc0287U, 0x3e80a1063d965d23U,
  0xbe1cc050bedbd586U, 0x3d8a5f883ebc3d8cU, 0xbe843649beca0ca1U, 0xbe1b9c9abe83954eU,
  0x3d82b9d03e3e61a2U, 0x3dc61ddcbe2a88d6U, 0xbc4b20e03e3ba6f2U, 0xbe7ebafa3ececd11U,
  0xbea57d5e3ed7c01bU, 0xbe966cff3edb899bU, 0xbe88e8dc3e993c1eU, 0x3e4e5732be69d74dU,
  0x3cd3fb503ebfc857U, 0xbd208b10be01b39aU, 0xbe9e49f0be0d1cf4U, 0xbd8f5bcf3eb0cac6U,
  0x3e59f3023cc4c1c0U, 0x3f421e26becaa804U, 0xbd817911beb4490bU, 0x3eafcb21be5187b1U,
  0xbe1d9c4f3dc0ff30U, 0xbeb0d870bdbde53aU, 0xbe57e1e63e6529d6U, 0xbf3cf0b23e3c918dU,
  0x3e71c84abeab75e0U, 0x3f0d1a00bf0acb62U, 0x3e7c08833e232459U, 0xbe38bf1c3e9de011U,
  0xbea382e3bd985bdfU, 0x3e8cb6dbbea45119U, 0xbeb16693bec76e0dU, 0xbcd330babfb5cc07U,
  0x3ea8e23f3d8991ccU, 0xbe090e5e3e99d1c5U, 0xbed9c640bd21f628U, 0x3ed48ac33e5d4fc2U,
  0x3e17839abe4952f5U, 0xbe742cfbbe16933aU, 0x3ea91373be9bfa09U, 0xbe45a6363e1ea2a2U,
  0xbeab3d873ea6f1b5U, 0xbe06bf0abe31538aU, 0xbeafb27ebe895a00U, 0x3e4194523ea64885U,
  0x3e86c82bbedcada0U, 0x3ea9d1a73e9608d3U, 0xbedb4efebd464d70U, 0x3ea686973e053112U,
  0x3d66c8283c6c3640U, 0x3eb29434bdb0e3b6U, 0xbe7f78533e8f4ce0U, 0x3e99b3a53afc1800U,
  0x3e90f4203e363522U, 0xbcb6e6503ea05392U, 0xbec7c6b23e2d8c1aU, 0xbe2bf5abbe09c7f2U,
  0x3db64a7c3e282f26U, 0xbebd82643d5a2230U, 0xbe096066bd0b5ab8U, 0x3e2227b6beafbdbcU,
  0x3b0a1e003e9b1077U, 0x3ed2db37be01a19cU, 0x3e08947a3e24d292U, 0xbe533251be575edbU,
  0x3e90db573eca3ff3U, 0x3e5c89583dc6adeeU, 0xbd57700f3e882acdU, 0xbe82226ebed7066dU,
  0x3e7abc2ebd7bccffU, 0x3b8588803eaa0217U, 0xbe0fa1683ece26bfU, 0xbc1dcc2cbcb458ceU,
  0x3cdf46203e4e47aeU, 0x3e1efd9e3e934671U, 0x3ddb4b3dbe80d0ebU, 0xbe8312823ebb1299U,
  0x3dc69d603eb8e224U, 0xbc1a7420be358272U, 0x3d88bfe43e8e281bU, 0x3d3c1f6d3e153be7U,
  0xbdeef554bd0ddfb0U, 0xbec122dabe8069a9U, 0xbc5b29b93eaac700U, 0xbe6d03dabdb96f14U,
  0x3e5641ec3e0780f1U, 0x3db9aa3cbe8adc25U, 0x3e8d32f33ed317c5U, 0x3e1bcdec3e17294bU,
  0x3e7aac1a3ed5f91fU, 0xbe7faba23e9075b4U, 0xbeca38123e1e170eU, 0xbd353390bedab2daU,
  0x3df3ba683e87740eU, 0x3cb7a190be340e41U, 0xbdfebf903edd508fU, 0x3dcad72bbe4a88c1U,
  0x3e47afbe3ea843b9U, 0xbe4f3e34be3e82ecU, 0xbda289a83d519358U, 0x3e161176bdac3e8cU,
  0x3e7ffeaabe9ab6f8U, 0xbe73a18dbd9b0d5cU, 0xbecfd25dbe9d204cU, 0xbdfc3bfc3eb8d56fU,
  0xbeb19e97bd82da4cU, 0xbc23a0503be9d287U, 0x3f015db5be0e2501U, 0x3f265254U,
  0x3d39a5d900000000U, 0x3d2ae48b00000000U, 0xbe2842bcbe6d57e7U, 0x3cc6e8b1U,
  0x3f62ac1dbd5f136eU, 0x3f092564bda2516bU, 0xbf84ef513e9dfb2cU, 0xbea40a3f3fbab0e5U,
  0xbe9aa28b3ed4238cU, 0xbec90d923f11b70cU, 0x3f2c0eb13ef450aaU, 0x3ef3dffcbeabc25fU,
  0xbe71d63eU,
};


ai_handle g_sine_model_weights_table[1 + 2] = {
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
  AI_HANDLE_PTR(s_sine_model_weights_array_u64),
  AI_HANDLE_PTR(AI_MAGIC_MARKER),
};

