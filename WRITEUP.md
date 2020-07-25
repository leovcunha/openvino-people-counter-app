# Project Write-Up

## Explaining Custom Layers

The process behind converting custom layers involves creating intermediate representation for layers not officially supported by OpenVINO.

To add them, extensions for both the Model Optimizer and Inference Engine are needed. The model extension generator that comes with OpenVINO generates template source files for each extension needed. The functions may need to be edited to create specialized extension source code. After that we use the Model Optimizer to convert and optimize the example TensorFlow model into IR files that will run inference using the Inference Engine.

There are a few different steps depending on the framework of origin. In TensorFlow for example an option is to register the custom layers as extensions to the Model Optimizer. Another option is to replace the unsupported subgraph with a different subgraph. A third option is to actually offload the computation of the subgraph back to TensorFlow during inference.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]

  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 2: [Name]

  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
