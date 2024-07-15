This repo hosts the official MediaPipe samples with a goal of showing the fundamental steps involved to create apps with our machine learning platform. 

External PRs for fixes are welcome, however new sample/demo PRs will likely be rejected to maintain the simplicity of this repo for ongoing maintenance. It is strongly recommended that contributors who are interested in submitting more complex samples or demos host their samples in their own public repos and create written tutorials to share with the community. Contributors can also submit these projects and tutorials to the [Google DevLibrary](https://devlibrary.withgoogle.com/)


MediaPipe Solutions streamlines on-device ML development and deployment with flexible low-code / no-code tools that provide the modular building blocks for creating custom high-performance solutions for cross-platform deployment. It consists of the following components:
* MediaPipe Tasks (low-code): create and deploy custom e2e ML solution pipelines
* MediaPipe Model Maker (low-code): create custom ML models from advanced solutions
* MediaPipe Studio (no-code): create, evaluate, debug, benchmark, prototype, deploy advanced production-level solutions

# MediaPipe LLM Inference Guide for Android

## Attention: Experimental API
The MediaPipe LLM Inference API is experimental and under active development. Usage is subject to the Generative AI Prohibited Use Policy.

## Overview
The LLM Inference API enables running large language models (LLMs) completely on-device for Android applications. It supports multiple text-to-text LLMs and can be used for tasks such as text generation, information retrieval, and document summarization. Supported models include Gemma 2B, Phi-2, Falcon-RW-1B, and StableLM-3B.

## Supported Models
- **Gemma 2B**: Lightweight, state-of-the-art open model for text generation tasks.
- **Phi-2**: 2.7 billion parameter Transformer model for Q&A, chat, and code format.
- **Falcon-RW-1B**: 1 billion parameter causal decoder-only model trained on 350B tokens of RefinedWeb.
- **StableLM-3B**: 3 billion parameter decoder-only language model pre-trained on 1 trillion tokens of diverse datasets.

## Setup Instructions

### Clone the Repository
To get started, clone the example code repository:

```sh
git clone https://github.com/google-ai-edge/mediapipe-samples
cd mediapipe
git sparse-checkout init --cone
git sparse-checkout set examples/llm_inference/android
```

### Dependencies
Add the following dependency to your `build.gradle` file:

```gradle
dependencies {
    implementation 'com.google.mediapipe:tasks-genai:0.10.14'
}
```

### Download and Convert Models
1. **Download Gemma 2B from Kaggle** (already compatible).
2. **Convert other models to MediaPipe format** using the following script:

```python
import mediapipe as mp
from mediapipe.tasks.python.genai import converter

config = converter.ConversionConfig(
  input_ckpt=INPUT_CKPT,
  ckpt_format=CKPT_FORMAT,
  model_type=MODEL_TYPE,
  backend=BACKEND,
  output_dir=OUTPUT_DIR,
  vocab_model_file=VOCAB_MODEL_FILE,
  output_tflite_file=OUTPUT_TFLITE_FILE,
)

converter.convert_checkpoint(config)
```

### Push Model to Device
```sh
adb shell rm -r /data/local/tmp/llm/ 
adb shell mkdir -p /data/local/tmp/llm/
adb push output_path /data/local/tmp/llm/model_version.bin
```

### Create the Task
Initialize the task with the following code:

```kotlin
val options = LlmInferenceOptions.builder()
        .setModelPath('/data/local/.../')
        .setMaxTokens(1000)
        .setTopK(40)
        .setTemperature(0.8)
        .setRandomSeed(101)
        .build()

llmInference = LlmInference.createFromOptions(context, options)
```

### Generate Response
To generate a response from the model:

```kotlin
val inputPrompt = "Compose an email to remind Brett of lunch plans at noon on Saturday."
val result = llmInference.generateResponse(inputPrompt)
logger.atInfo().log("result: $result")
```

For asynchronous generation:

```kotlin
val options = LlmInference.LlmInferenceOptions.builder()
  .setResultListener { partialResult, done ->
    logger.atInfo().log("partial result: $partialResult")
  }
  .build()

llmInference.generateResponseAsync(inputPrompt)
```

### Handle Results
The `LLM Inference API` returns a `LlmInferenceResult` containing the generated response text.

## LoRA Model Customization
The API supports Low-Rank Adaptation (LoRA) for large language models. This enables cost-effective training for model customization.

### Prepare LoRA Models
Train a LoRA model on your dataset using the instructions on HuggingFace, then convert it to a TensorFlow Lite Flatbuffer:

```python
import mediapipe as mp
from mediapipe.tasks.python.genai import converter

config = converter.ConversionConfig(
  backend='gpu',
  lora_ckpt=LORA_CKPT,
  lora_rank=LORA_RANK,
  lora_output_tflite_file=LORA_OUTPUT_TFLITE_FILE,
)

converter.convert_checkpoint(config)
```

### Initialize with LoRA Model
```kotlin
val options = LlmInferenceOptions.builder()
        .setModelPath('<path to base model>')
        .setMaxTokens(1000)
        .setTopK(40)
        .setTemperature(0.8)
        .setRandomSeed(101)
        .setLoraPath('<path to LoRA model>')
        .build()

llmInference = LlmInference.createFromOptions(context, options)
```

## Training the Model on Google Colab
To train the model on Google Colab, use the following steps:

1. Install required dependencies:
    ```sh
    !pip install mediapipe
    ```
2. Follow the training instructions on HuggingFace to fine-tune your model.
3. Use the provided conversion script to convert and bundle your trained model.

## Building the APK in Android Studio
1. **Import the Project**: Open Android Studio and select "Open an Existing Project" to import the cloned repository.
2. **Build the APK**: Navigate to `Build > Build Bundle(s) / APK(s) > Build APK(s)` to create the APK.
3. **Run on Device**: Connect your Android device via USB, select it as the target device, and click "Run".

## Conclusion
This guide provides a comprehensive overview of setting up and using the MediaPipe LLM Inference API on Android. By following these steps, you can integrate state-of-the-art LLMs into your Android applications for various text generation tasks.

For more details and updates, refer to the [official documentation](https://google.github.io/mediapipe/).

---

For any issues or contributions, please refer to the [GitHub repository](https://github.com/google-ai-edge/mediapipe-samples).
