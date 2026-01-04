# AraEduBench

**AraEduBench** is a benchmark for evaluating large language models on **Arabic educational scenarios**.  
It follows an **evaluation-first** design and focuses on how models behave in realistic educational contexts, rather than only measuring factual accuracy.

This repository provides the dataset files and documentation to support research on Arabic educational language models.

---

## Overview

AraEduBench is designed to evaluate large language models across a diverse set of educational tasks in Arabic. The benchmark covers both **student-oriented** and **teacher-oriented** scenarios and spans multiple educational levels and subject areas.

Key characteristics:
- Fully Arabic-native
- Scenario-based design
- Evaluation-first benchmarking
- Covers general and higher education
- Publicly released for research use

---

## Educational Scenarios

AraEduBench includes **nine educational scenarios**, grouped into two main categories.

### Student-Oriented Scenarios
- Problem Solving (Q&A)
- Error Correction
- Idea Provision
- Personalized Learning Support
- Emotional Support

### Teacher-Oriented Scenarios
- Question Generation
- Automatic Grading
- Teaching Material Generation
- Personalized Content Creation

Each scenario is provided as a **separate JSONL file**.

---

## Dataset Structure

- Total files: **9 JSONL files**
- Total records: **9,000+**
- Each file corresponds to one scenario
- Records are generated using fixed templates with validation rules

Each record includes a shared set of mandatory fields, such as:
- `Subject`
- `Level`
- `Question Type`
- Scenario-specific question content
- Model response fields (used during evaluation)

The exact fields may vary slightly by scenario to reflect task-specific requirements.

---

## Educational Coverage

### Education Levels
- General Education: Elementary, Middle School, High School
- Higher Education: Bachelor’s, Master’s, Doctorate

### Subjects
- General education subjects include Arabic, Mathematics, English, Physics, Chemistry, Biology, History, and Geography.
- Higher education subjects span multiple domains, including sciences, engineering, economics, medicine, social sciences, and military studies.

The dataset uses **generic Arabic educational content** and does not target any country-specific curriculum.

---

## Language

- All content is written in **Modern Standard Arabic**
- Arabic-only generation is explicitly enforced
- No dialectal Arabic is included in the current version

---

## Evaluation

AraEduBench is designed for **evaluation-first benchmarking**.

Evaluation is based on:
- Scenario-aware metric allocation
- Three metric categories:
  - Scenario Adaptation
  - Content Accuracy
  - Pedagogical Effectiveness
- Twelve sub-metrics in total

The benchmark supports:
- LLM-based evaluation
- Human evaluation
- Anonymized model comparison to reduce evaluation bias

Full evaluation details are described in the accompanying paper.

---

## Intended Use

This dataset is intended for:
- Research on Arabic educational language models
- Benchmarking and evaluation studies
- Analysis of pedagogical behavior in LLMs

It is **not intended** for:
- Real-world student grading
- Deployment in educational decision-making systems

---
