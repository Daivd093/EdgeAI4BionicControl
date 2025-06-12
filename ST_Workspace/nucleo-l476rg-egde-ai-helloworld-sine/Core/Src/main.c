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

#include "ai_datatypes_defines.h"
#include "ai_platform.h"
#include "sine_model.h"
#include "sine_model_data.h"


/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define RX_DATA_SIZE 8 // Cantidad de dígitos (+1) de los números de entrada
#define VERBOSE 1   // Con VERBOSE 0 solo quedan mensajes de error y se entregan (IN, OUT)

//Parámetros de escala
//#define SCALE_IN  0.024420151486992836f
//#define ZERO_IN  (-128)
//#define SCALE_OUT  0.0082210972905159f
//#define ZERO_OUT   (5)


/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
CRC_HandleTypeDef hcrc;

TIM_HandleTypeDef htim16;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

uint8_t rx_indx;
uint8_t rx_char;
char rx_data[RX_DATA_SIZE]; // Pensado en números entre 0 y 2*pi con RX_DATA_SIZE-3 decimales. Ej: RX_DATA_SIZE = 8 -> "3.14159\0"


/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_CRC_Init(void);
static void MX_TIM16_Init(void);
/* USER CODE BEGIN PFP */

//static inline int8_t quantize(float x);
//static inline float dequantize(int8_t q);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/*
static inline int8_t quantize(float x){

	float qf = x / SCALE_IN + (float)ZERO_IN;
	if (qf > 127.0f)  qf = 127.0f;
	if (qf < -128.0f) qf = -128.0f;
	return (int8_t)qf;
}

static inline float dequantize(int8_t q){

	float dq = ((float)q - (float)ZERO_OUT)*SCALE_OUT;
	return dq;
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
    char buf[50];
    int buf_len = 0;
    ai_error ai_err;
    ai_i32 nbatch;
    uint32_t timestamp;
    float y_val;

    // Espacio de memoria para guardar calculos intermedios (usando variables definidas por Cube.IA, no por mi)
    AI_ALIGNED(4) ai_u8 activations[AI_SINE_MODEL_DATA_ACTIVATIONS_SIZE];

    // Buffers para los tensores de entradas y salidas
    AI_ALIGNED(4) ai_float in_data[AI_SINE_MODEL_IN_1_SIZE]; // Esto espera flotante de 32 bits
    AI_ALIGNED(4) ai_float out_data[AI_SINE_MODEL_OUT_1_SIZE];
    //AI_ALIGNED(4) ai_i8 in_data[AI_SINE_MODEL_IN_1_SIZE]; // Esto espera entero de 8 bits con signo
	//AI_ALIGNED(4) ai_i8 out_data[AI_SINE_MODEL_OUT_1_SIZE];


    ai_handle sine_model = AI_HANDLE_NULL;

    // Para guardar punteros hacia los datos
    ai_buffer ai_input[AI_SINE_MODEL_IN_NUM];
    ai_buffer ai_output[AI_SINE_MODEL_OUT_NUM];

    ai_shape_dimension input_shape_data[4] = {1, 1, 1, 1};
	ai_shape_dimension output_shape_data[4] = {1, 1, 1, 1};



    // Set working memory and get weights/bias from model
	ai_network_params ai_params = {
	    .params = AI_SINE_MODEL_DATA_WEIGHTS(ai_sine_model_data_weights_get()),
	    .activations = AI_SINE_MODEL_DATA_ACTIVATIONS(activations),
	};

    // Pointer wrapper structs to data buffers
    ai_input[0].data = AI_HANDLE_PTR(in_data);
    ai_input[0].shape.size = 4;
    ai_input[0].shape.data = input_shape_data;
    ai_input[0].format = AI_BUFFER_FORMAT_FLOAT;
    //ai_input[0].format = AI_BUFFER_FORMAT_Q7;


    ai_output[0].data = AI_HANDLE_PTR(out_data);
	ai_output[0].shape.size = 4;
	ai_output[0].shape.data = output_shape_data;
	ai_output[0].format = AI_BUFFER_FORMAT_FLOAT;
	//ai_output[0].format = AI_BUFFER_FORMAT_Q7;

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
  MX_USART2_UART_Init();
  MX_CRC_Init();
  MX_TIM16_Init();
  /* USER CODE BEGIN 2 */

  // Start timer/counter
  HAL_TIM_Base_Start(&htim16);
  if (VERBOSE){
	  // Saludo
	  buf_len = sprintf(buf, "\r\n\r\nSTM32 X-Cube-AI test\r\n");
	  HAL_UART_Transmit(&huart2, (uint8_t * )buf, buf_len, 100);
  }

  // La red en sí
  ai_err = ai_sine_model_create(&sine_model, AI_SINE_MODEL_DATA_CONFIG);
  if (ai_err.type != AI_ERROR_NONE)
  {
	  buf_len = sprintf(buf, "Error: could not create NN instance\r\n");
	  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	  while(1);
  }

  // Inicializar la NN
  if (!ai_sine_model_init(sine_model, &ai_params))
  {
	  buf_len = sprintf(buf, "Error, could not initialize NN\r\n");
	  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	  while(1);
  }





  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */

  while (1)
  {
	  // Fill input buffer



	  //uint8_t num;
	  float user_input = 0.0f;

	  if (VERBOSE){
		  HAL_UART_Transmit(&huart2, (uint8_t *)"\r\nIngresa un número positivo menor a 2 pi tienes 7 caracteres:\r\n", 65, 100);

	  }

	  // Receive bloqueante

	  for(rx_indx = 0; rx_indx < RX_DATA_SIZE-1 ; rx_indx++){
	  	  HAL_UART_Receive(&huart2, &rx_char, 1, HAL_MAX_DELAY); //Recibe 1 caracter

	  	  if(VERBOSE){
		  	  HAL_UART_Transmit(&huart2, &rx_char, 1, 10);
	  	  }

		  if(rx_char == '\r' || rx_char == '\n') break;

		  if ((rx_char >= '0' && rx_char <= '9') || rx_char == '.' || (rx_char == '-' && rx_indx == 0 )){
			  rx_data[rx_indx]=rx_char;
		  }
	  }
  	  rx_data[rx_indx] = '\0';

  	  if (VERBOSE){
		  // Echo Final
		  //HAL_UART_Transmit(&huart2, rx_data, rx_indx, 10);
		  HAL_UART_Transmit(&huart2, (uint8_t *)"\r\n", 2, 10);
  	  }

	  sscanf(rx_data, "%f", &user_input);
	  if (VERBOSE){
		  if ( user_input < 0 || user_input > 6.28) {
			  HAL_UART_Transmit(&huart2, (uint8_t *)"Los resultados no serán satisfactorios\r\n", 42, 100);
		  }
	  }
	  for (uint32_t i = 0; i < AI_SINE_MODEL_IN_1_SIZE; i++)
	  {
		  in_data[i] = user_input;//quantize(user_input);
	  }

	  // timestamp inicio
	  timestamp = htim16.Instance->CNT;

	  // Realizar inferencia
	  nbatch = ai_sine_model_run(sine_model, &ai_input[0], &ai_output[0]);
	  if (nbatch != 1)
	    {
	  	  buf_len = sprintf(buf, "Error, could not run inference\r\n");
	  	  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);
	  	  while(1);
	    }
	  long unsigned int tim = htim16.Instance->CNT - timestamp;
	 // Lee la prediccion
	  y_val = out_data[0];//dequantize(out_data[0]);
	  if (VERBOSE){
		  buf_len = sprintf(buf, "Input: %.3f -> Output: %.5f | Tiempo: %lu us\r\n", user_input, y_val, tim);
	  } else {
		  buf_len = sprintf(buf, "(%.5f, %.5f)\r\n", user_input, y_val);
	  }
	  HAL_UART_Transmit(&huart2, (uint8_t *)buf, buf_len, 100);

	  //HAL_Delay(500);

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
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
  htim16.Init.Prescaler = 80 - 1;
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


//void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
//{
//  /* Prevent unused argument(s) compilation warning */
//  UNUSED(huart);
//
//  /* NOTE : This function should not be modified, when the callback is needed,
//            the HAL_UART_RxCpltCallback can be implemented in the user file.
//   */
//  HAL_UART_Transmit(&huart2, rx_data, 1, 10);
//
//  HAL_UART_Receive_IT(&huart2, rx_data, 1);
//
//}


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
