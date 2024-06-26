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

#ifndef NNETWORK__NNETWORK_HPP_
#define NNETWORK__NNETWORK_HPP_

#include <cstdint>

#include "nnetwork/visibility_control.hpp"


namespace nnetwork
{

class NNETWORK_PUBLIC Nnetwork
{
public:
  Nnetwork();
  int64_t foo(int64_t bar) const;
};

}  // namespace nnetwork

#endif  // NNETWORK__NNETWORK_HPP_
