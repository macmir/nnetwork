// Copyright 2024 111
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NNETWORK__VISIBILITY_CONTROL_HPP_
#define NNETWORK__VISIBILITY_CONTROL_HPP_

////////////////////////////////////////////////////////////////////////////////
#if defined(__WIN32)
  #if defined(NNETWORK_BUILDING_DLL) || defined(NNETWORK_EXPORTS)
    #define NNETWORK_PUBLIC __declspec(dllexport)
    #define NNETWORK_LOCAL
  #else  // defined(NNETWORK_BUILDING_DLL) || defined(NNETWORK_EXPORTS)
    #define NNETWORK_PUBLIC __declspec(dllimport)
    #define NNETWORK_LOCAL
  #endif  // defined(NNETWORK_BUILDING_DLL) || defined(NNETWORK_EXPORTS)
#elif defined(__linux__)
  #define NNETWORK_PUBLIC __attribute__((visibility("default")))
  #define NNETWORK_LOCAL __attribute__((visibility("hidden")))
#elif defined(__APPLE__)
  #define NNETWORK_PUBLIC __attribute__((visibility("default")))
  #define NNETWORK_LOCAL __attribute__((visibility("hidden")))
#else
  #error "Unsupported Build Configuration"
#endif

#endif  // NNETWORK__VISIBILITY_CONTROL_HPP_
