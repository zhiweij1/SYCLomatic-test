/*
Modifications Copyright (C) 2023 Intel Corporation

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


SPDX-License-Identifier: BSD-3-Clause
*/

/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef MC_PARTICLE_INCLUDE
#define MC_PARTICLE_INCLUDE

#include "DirectionCosine.hh"
#include "Tallies.hh"

#include "MC_Vector.hh"
#include "MC_Facet_Adjacency.hh"
#include "MC_Location.hh"

#include "DeclareMacro.hh"

class MC_Base_Particle;

class MC_Particle
{
public:
    // the current position of the particle
    MC_Vector coordinate;

    // the velocity of the particle
    MC_Vector velocity;

    // the direction of the particle
    DirectionCosine direction_cosine;

    // the kinetic energy of the particle
    double kinetic_energy;

    // the weight of the particle
    double weight;

    // the time remaining for this particle to hit census
    double time_to_census;

    // cacheing the current total cross section
    double totalCrossSection;

    // the age of this particle
    double age;

    // the number of mean free paths to a collision
    double num_mean_free_paths;

    // distance to a collision
    double mean_free_path;

    // the distance this particle travels in a segment.
    double segment_path_length;

    // the random number seed for the random number generator for this particle
    uint64_t random_number_seed;

    // unique identifier used to identify and track individual particles in the simulation
    uint64_t identifier;

    // the last event this particle underwent
    MC_Tally_Event::Enum last_event;

    int num_collisions;

    double num_segments;

    // task working on, used to index into
    int task;

    // species of the particle
    int species;

    // the breed of the particle how it was produced
    int breed;

    // current energy group of the particle
    int energy_group;

    // its current domain in the spatial decomposition
    int domain;

    // the current cell in its current domain
    int cell;

    int facet;

    // When crossing a facet, keep the surface normal dot product
    double normal_dot;

public:
    HOST_DEVICE_CUDA
    MC_Particle();

    HOST_DEVICE_CUDA
    MC_Particle(const MC_Base_Particle &from_particle);

    HOST_DEVICE_CUDA
    void Copy_From_Base(const MC_Base_Particle &from_particle);

    HOST_DEVICE_CUDA
    MC_Location Get_Location() const;

    // format a string with the contents of the particle
    void Copy_Particle_To_String(std::string &output_string) const;

    // move a particle a distance in the direction_cosine direction
    HOST_DEVICE_CUDA
    void Move_Particle(const DirectionCosine &direction_cosine, const double distance);

    HOST_DEVICE_CUDA
    void PrintParticle();

    HOST_DEVICE_CUDA
    DirectionCosine *Get_Direction_Cosine()
    {
        return &this->direction_cosine;
    }

    HOST_DEVICE_CUDA
    MC_Vector *Get_Velocity()
    {
        return &this->velocity;
    }
};

//----------------------------------------------------------------------------------------------------------------------
//  Return a MC_Location given domain, cell, facet.
//----------------------------------------------------------------------------------------------------------------------
HOST_DEVICE_CUDA
inline MC_Location MC_Particle::Get_Location() const
{
    return MC_Location(domain, cell, facet);
}

//----------------------------------------------------------------------------------------------------------------------
//  Move the particle a straight-line distance along a specified cosine.
//----------------------------------------------------------------------------------------------------------------------
HOST_DEVICE_CUDA
inline void MC_Particle::Move_Particle(const DirectionCosine &my_direction_cosine,
                                       const double distance)
{
    coordinate.x += (my_direction_cosine.alpha * distance);
    coordinate.y += (my_direction_cosine.beta * distance);
    coordinate.z += (my_direction_cosine.gamma * distance);
}

//----------------------------------------------------------------------------------------------------------------------
//  Print all of the particles components
//----------------------------------------------------------------------------------------------------------------------
HOST_DEVICE_CUDA
inline void MC_Particle::PrintParticle()
{
    /*
    DPCT1040:0: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("coordiante:          %g\t%g\t%g\n", coordinate.x, coordinate.y,
           coordinate.z);
    /*
    DPCT1040:1: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("velocity:            %g\t%g\t%g\n", velocity.x, velocity.y,
           velocity.z);
    /*
    DPCT1040:2: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("direction_cosine:     %g\t%g\t%g\n", direction_cosine.alpha,
           direction_cosine.beta, direction_cosine.gamma);
    /*
    DPCT1040:3: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("kinetic_energy:      %g\n", kinetic_energy);
    /*
    DPCT1040:4: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("Weight:              %g\n", weight);
    /*
    DPCT1040:5: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("time_to_census:      %g\n", time_to_census);
    /*
    DPCT1040:6: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("totalCrossSection:   %g\n", totalCrossSection);
    /*
    DPCT1040:7: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("age:                 %g\n", age);
    /*
    DPCT1040:8: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("num_mean_free_paths: %g\n", num_mean_free_paths);
    /*
    DPCT1040:9: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("mean_free_path:      %g\n", mean_free_path);
    /*
    DPCT1040:10: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("segment_path_length: %g\n", segment_path_length);
    /*
    DPCT1040:11: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("random_number_seed:  %zu\n", random_number_seed);
    /*
    DPCT1040:12: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("identifier:          %zu\n", identifier);
    /*
    DPCT1040:13: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("last_event:          %d\n", last_event);
    /*
    DPCT1040:14: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("num_collision:       %d\n", num_collisions);
    /*
    DPCT1040:15: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("num_segments:        %g\n", num_segments);
    /*
    DPCT1040:16: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("task:                %d\n", task);
    /*
    DPCT1040:17: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("species:             %d\n", species);
    /*
    DPCT1040:18: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("breed:               %d\n", breed);
    /*
    DPCT1040:19: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("energy_group:        %d\n", energy_group);
    /*
    DPCT1040:20: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("domain:              %d\n", domain);
    /*
    DPCT1040:21: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("cell:                %d\n", cell);
    /*
    DPCT1040:22: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("facet:               %d\n", facet);
    /*
    DPCT1040:23: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("normal_dot:          %g\n", normal_dot);
    /*
    DPCT1040:24: Use sycl::stream instead of printf if your code is used on the
    device.
    */
    printf("\n");
}

int MC_Copy_Particle_Get_Num_Fields();

#endif //  MC_PARTICLE_INCLUDE
