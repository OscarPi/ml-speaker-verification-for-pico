/*
 * Copyright (c) 2021 Arm Limited and Contributors. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 * 
 */

#include <stdio.h>
#include "pico/stdlib.h"
#include "tflite_model.h"
#include "ml_model.h"
#include <inttypes.h>

// constants
#define TENSOR_ARENA_SIZE 190000

MLModel ml_model(tflite_model, TENSOR_ARENA_SIZE);

int main( void )
{
    stdio_init_all();

    printf("Start of main...\n");

    if (!ml_model.init()) {
        printf("Failed to initialize ML model!\n");
        while (1) { tight_loop_contents(); }
    }

    uint64_t start;
    uint32_t time_taken;

    start=time_us_32();
    float prediction = ml_model.predict();
    time_taken = time_us_32() - start;

    printf("Ran ML model: time taken %"PRIu32"\n", time_taken);

    return 0;
}

