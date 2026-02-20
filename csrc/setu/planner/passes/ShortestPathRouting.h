#pragma once

#include "planner/ir/cir/Program.h"
#include "planner/topo/Topology.h"

namespace setu::planner::passes {
namespace cir = setu::planner::ir::cir;
using setu::planner::topo::TopologyPtr;

class ShortestPathRouting {
 public:
  explicit ShortestPathRouting(TopologyPtr topo) : topo_(std::move(topo)) {}
  cir::Program Run(const cir::Program& program);

 private:
  TopologyPtr topo_;
};
}  // namespace setu::planner::passes
