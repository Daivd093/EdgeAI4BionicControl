/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include <stdio.h>
#include <stdint.h>  // Para evitar advertencias de int8 o uint8
#include <string.h> // Para usar memcpy

#include "ai_datatypes_defines.h"
#include "ai_platform.h"
#include "model_s2f1.h"
#include "model_s2f1_data.h"
//#include "model_s2f2.h"
//#include "model_s2f2_data.h"
//#include "model_s2f3.h"
//#include "model_s2f3_data.h"
//#include "model_s2f4.h"
//#include "model_s2f4_data.h"
//#include "model_s2f5.h"
//#include "model_s2f5_data.h"



/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#define VERBOSE 0   // Con VERBOSE 0 solo quedan mensajes de error y se entregan (IN, OUT)
//#define NUM_INPUTS 1153 // Cantidad de columnas en la matriz R. Por sujeto las opciones son: 1489/1153/1537
#define INPUT_BITS ( 4 * AI_MODEL_S2F1_IN_1_SIZE ) // Son float32 los datos de entrada
#define NUM_FINGERS 1 // Idealmente la idea es llegar a 5

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CRC_HandleTypeDef hcrc;

TIM_HandleTypeDef htim16;

UART_HandleTypeDef huart2;
DMA_HandleTypeDef hdma_usart2_rx;

/* USER CODE BEGIN PV */

//uint8_t TxData[(4 * NUM_FINGERS)];
AI_ALIGNED(4) uint8_t RxData[(INPUT_BITS+4)]; // Alineado usando la macro de "ai_platform.h"
//int isSizeRxed = 0;

uint8_t RxSize[4];
uint32_t size = 0;


/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_TIM16_Init(void);
static void MX_CRC_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/*
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart){

	if (isSizeRxed == 0){ // Si no he recibido el tamaño de los datos que recibiré
		// Los primeros 4 bytes que reciba serán el tamaño
		size = ((RxData[0]-48)*1000)+((RxData[1]-48)*100)+((RxData[2]-48)*10)+((RxData[3]-48));
		// Le resto 48 para pasar de ascii al número.
		isSizeRxed = 1;
		HAL_UART_Receive_DMA(&huart2, RxData,size);

	}
	else if (isSizeRxed == 1){
		isSizeRxed = 0;

		for (int i = 0; i < AI_MODEL_S2F1_IN_1_SIZE; ++i) {
		    memcpy(&in_data_s2f1[i], &RxData[i * 4], sizeof(float));
		}

		HAL_UART_Receive_DMA(&huart2, RxData,4);

	}

}
*/


/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

	char buf[60];
	int buf_len = 0;
	ai_error ai_err;
	ai_i32 nbatch;
	// uint32_t timestamp;
	float y_val_s2f1;

	// Espacio de memoria para guardar calculos intermedios (usando variables definidas por Cube.IA, no por mi)
	AI_ALIGNED(4) ai_u8 activations_s2f1[AI_MODEL_S2F1_DATA_ACTIVATIONS_SIZE];

	// Buffers para los tensores de entradas y salidas
	AI_ALIGNED(4) ai_float in_data_s2f1[AI_MODEL_S2F1_IN_1_SIZE]; // Esto espera 1153 flotantes de 32 bits
	AI_ALIGNED(4) ai_float out_data_s2f1[AI_MODEL_S2F1_IN_1_SIZE];

	ai_handle model_s2f1 = AI_HANDLE_NULL;

	// Para guardar punteros hacia los datos
	ai_buffer ai_input_s2f1[AI_MODEL_S2F1_IN_NUM];
	ai_buffer ai_output_s2f1[AI_MODEL_S2F1_OUT_NUM];



	// {N,H,W,C} : Num_Muestras, Altura, Ancho, Canales
	ai_shape_dimension input_shape_data_s2f1[4] =  {1, AI_MODEL_S2F1_IN_1_SIZE, 1, 1};  	// 1 muestra de algo de 1153x1, de 1 canal
	ai_shape_dimension output_shape_data_s2f1[4] = {1, 1, 1, 1};  		// 1 escalar (1 muestra de 1x1 de 1 canal)


	// Set working memory and get weights/bias from model
	ai_network_params ai_params_s2f1 = {
		.params = AI_MODEL_S2F1_DATA_WEIGHTS(ai_model_s2f1_data_weights_get()),  // En realidad esta forma de hacerlo está obsoleta
		.activations = AI_MODEL_S2F1_DATA_ACTIVATIONS(activations_s2f1),		 // deprecated
	};																			 // Luego revisaré cómo se hace actualmente

	// Pointer wrapper structs to data buffers
	ai_input_s2f1[0].data = AI_HANDLE_PTR(in_data_s2f1);
	ai_input_s2f1[0].shape.size = 4;
	ai_input_s2f1[0].shape.data = input_shape_data_s2f1;
	ai_input_s2f1[0].format = AI_BUFFER_FORMAT_FLOAT;

	ai_output_s2f1[0].data = AI_HANDLE_PTR(out_data_s2f1);
	ai_output_s2f1[0].shape.size = 4;
	ai_output_s2f1[0].shape.data = output_shape_data_s2f1;
	ai_output_s2f1[0].format = AI_BUFFER_FORMAT_FLOAT;



  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_USART2_UART_Init();
  MX_TIM16_Init();
  MX_CRC_Init();
  /* USER CODE BEGIN 2 */


  // Start timer/counter
  HAL_TIM_Base_Start(&htim16);
  if (VERBOSE){
	  // Saludo
	  buf_len = sprintf(buf, "\r\n\r\nSTM32 X-Cube-AI 1 Finger Test\r\n");
	  HAL_UART_Transmit(&huart2, (uint8_t * )buf, buf_len, 100);
  }

  // La red en sí
  ai_err = ai_model_s2f1_create(&model_s2f1, AI_MODEL_S2F1_DATA_CONFIG);
  if (ai_err.type != AI_ERROR_NONE)
  {
	  buf_len = sprintf(buf, "Error: could not create NN instance\r\n");
	  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	  while(1);
  }


  if (VERBOSE){
	  ai_network_report report;
	  // Revisando las dimensiones reales de entrada y salida
	  if (ai_model_s2f1_get_info(model_s2f1, &report)){
		  printf("Input shape (dims=%d): ", report.inputs->shape.size);
			  for (int i = 0; i < report.inputs->shape.size; ++i)
				  printf("%lu ", (unsigned long)report.inputs->shape.data[i]);
			  printf("\r\n");
		  printf("Output shape (dims=%d): ", report.outputs->shape.size);
			  for (int i = 0; i < report.outputs->shape.size; ++i)
				  printf("%lu ", (unsigned long)report.outputs->shape.data[i]);
			  printf("\r\n");
	  }
  }



  // Inicializar la NN
  if (!ai_model_s2f1_init(model_s2f1, &ai_params_s2f1))
  {
	  buf_len = sprintf(buf, "Error, could not initialize NN\r\n");
	  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	  while(1);
  }



  //HAL_UART_Receive_DMA(&huart2, RxData,4); // La primera vez espero 4 bytes con el número de 4 dígitos que será el tamaño a recibir


  // Voy a leer 1 sola vez el tamaño y luego esperar que se mande esa cantidad de números varias veces.
  // Si se quiere cambiar el tamaño, habrá que reiniciar
  if (VERBOSE){
  	 	  // Esperando
  	 	  buf_len = sprintf(buf, "Introduzca 4 dígitos con la cantidad de bytes a enviar\r\n");
  	 	  HAL_UART_Transmit(&huart2, (uint8_t * )buf, buf_len, 100);
  	   }

  	  HAL_UART_Receive(&huart2,RxSize,4,HAL_MAX_DELAY);
  	  // Los primeros 4 bytes que reciba serán el tamaño
  	  size = ((RxSize[0]-48)*1000)+((RxSize[1]-48)*100)+((RxSize[2]-48)*10)+((RxSize[3]-48));
  	  // Le resto 48 para pasar de ascii al número.


  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {

	  // timestamp inicio
	  //timestamp = htim16.Instance->CNT;

	  if (VERBOSE){
		  buf_len = sprintf(buf, "Esperando a recibir un grupo de %ld bytes\r\n", size);
		  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	  }

	  HAL_UART_Receive(&huart2, RxData,size,HAL_MAX_DELAY);

	  memcpy(in_data_s2f1, RxData, sizeof(float) * AI_MODEL_S2F1_IN_1_SIZE);

	  if (VERBOSE){
		  for (int i = 0; i < AI_MODEL_S2F1_IN_1_SIZE; ++i) {
		    buf_len = sprintf(buf, "in_data_s2f1[%d] = %.5f \r\n",i,in_data_s2f1[i]);
			HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
		}
	  }
	  if (VERBOSE){
		  buf_len = sprintf(buf, "Ya se convirtieron a flotantes\r\n");
		  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	  }

	  // Realizar inferencia
	  nbatch = ai_model_s2f1_run(model_s2f1, &ai_input_s2f1[0], &ai_output_s2f1[0]);
	  if (nbatch != 1)
	    {
	  	  buf_len = sprintf(buf, "Error, could not run inference\r\n");
	  	  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	  	  while(1);
	    }
	  //long unsigned int tim = htim16.Instance->CNT - timestamp;
	 // Lee la prediccion
	  y_val_s2f1 = out_data_s2f1[0];
	  if (VERBOSE){
		  buf_len = sprintf(buf, "Resultado = %.5f\r\n", y_val_s2f1);
	  } else {
		  buf_len = sprintf(buf,"%.8f\r\n",y_val_s2f1);

	 }
	 HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);


    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	//  		  HAL_UART_Transmit(&huart2, TxData,10240, HAL_MAX_DELAY);



	 //HAL_Delay(500);

  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 10;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void)
{

  /* USER CODE BEGIN CRC_Init 0 */

  /* USER CODE END CRC_Init 0 */

  /* USER CODE BEGIN CRC_Init 1 */

  /* USER CODE END CRC_Init 1 */
  hcrc.Instance = CRC;
  hcrc.Init.DefaultPolynomialUse = DEFAULT_POLYNOMIAL_ENABLE;
  hcrc.Init.DefaultInitValueUse = DEFAULT_INIT_VALUE_ENABLE;
  hcrc.Init.InputDataInversionMode = CRC_INPUTDATA_INVERSION_NONE;
  hcrc.Init.OutputDataInversionMode = CRC_OUTPUTDATA_INVERSION_DISABLE;
  hcrc.InputDataFormat = CRC_INPUTDATA_FORMAT_BYTES;
  if (HAL_CRC_Init(&hcrc) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN CRC_Init 2 */

  /* USER CODE END CRC_Init 2 */

}

/**
  * @brief TIM16 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM16_Init(void)
{

  /* USER CODE BEGIN TIM16_Init 0 */

  /* USER CODE END TIM16_Init 0 */

  /* USER CODE BEGIN TIM16_Init 1 */

  /* USER CODE END TIM16_Init 1 */
  htim16.Instance = TIM16;
  htim16.Init.Prescaler = 80-1;
  htim16.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim16.Init.Period = 65535;
  htim16.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim16.Init.RepetitionCounter = 0;
  htim16.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim16) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM16_Init 2 */

  /* USER CODE END TIM16_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Channel6_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Channel6_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Channel6_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */


// Redireccion de printf para evitar errores en Build

int __io_putchar(int ch)
{
    HAL_UART_Transmit(&huart2, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch;
}

int _write(int file, char *ptr, int len)
{
    HAL_UART_Transmit(&huart2, (uint8_t *)ptr, len, HAL_MAX_DELAY);
    return len;
}


// Stubs para que el linker deje de llorar
int _close(int file) { return -1; }
int _fstat(int file, void *st) { return 0; }
int _isatty(int file) { return 1; }
int _lseek(int file, int ptr, int dir) { return 0; }
int _read(int file, char *ptr, int len) { return 0; }
int _kill(int pid, int sig) { return -1; }
int _getpid(void) { return 1; }



/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
