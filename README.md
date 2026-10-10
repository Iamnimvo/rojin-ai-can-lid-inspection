![Uploading image.png…]()


# Rojin AI Can Lid Inspection

## AI-Based Visual Quality Inspection System for Tomato Paste Can Lids

This project presents a deep learning-based computer vision proof of
concept for automated quality inspection of tomato paste can lids.

The goal is to demonstrate the feasibility of using artificial
intelligence and image processing techniques to classify lid images into
acceptable and defective categories before moving toward a
factory-specific industrial deployment.

> This repository represents an offline Proof of Concept (PoC) based on
> public/external lid image data. Production deployment requires
> validation using Rojin factory images and industrial hardware
> integration.

## Project Overview

The current AI inspection pipeline includes:

-   Lid image preprocessing
-   Circular lid region extraction
-   Mask generation
-   Deep learning model training
-   Model evaluation
-   Probability-based decision logic
-   Result reporting

## System Pipeline

    Input Image
        |
        v
    Image Preprocessing
        |
        v
    Lid ROI Extraction
        |
        v
    Mask Generation
        |
        v
    YOLO11n Classification Model
        |
        v
    Probability Analysis
        |
        v
    GOOD / REVIEW / DEFECTIVE Decision

## Model Information

-   Model: YOLO11n Classification
-   Framework: Ultralytics YOLO
-   Training approach: Transfer Learning
-   Input size: 640 px

Classes:

-   GOOD
-   DEFECTIVE

A separate decision layer applies confidence thresholds to generate:

-   GOOD
-   REVIEW
-   DEFECTIVE

## Dataset

Dataset:

TinplateLidDATA2024

-   Source: Public external dataset
-   Total images: 1,307

## Results

Offline evaluation:

-   Test images: 261
-   Correct operational decisions: 257 / 261
-   Decision accuracy: 98.47%
-   Defective lid detection: 100% (147/147)

These results represent performance on the public dataset and do not
represent production-line accuracy.

## Repository Structure

    rojin-ai-can-lid-inspection/

    ├── src/
    ├── results/
    ├── requirements.txt
    ├── Proposal.pdf
    └── Handoff Documentation.pdf

## Technologies

-   Python
-   OpenCV
-   PyTorch
-   Ultralytics YOLO
-   Computer Vision
-   Deep Learning
-   Industrial AI

## Current Scope

Implemented:

-   Offline lid image classification
-   Image preprocessing pipeline
-   Model training and evaluation
-   Decision threshold logic
-   Performance reporting

Future development:

-   Industrial camera integration
-   Real-time conveyor inspection
-   PLC communication
-   Reject mechanism
-   Factory-specific validation

## Project Status

Proof of Concept (PoC)

Technical feasibility demonstration for AI-based visual inspection of
tomato paste can lids.
