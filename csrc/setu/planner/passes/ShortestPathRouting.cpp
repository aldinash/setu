#include "planner/passes/ShortestPathRouting.h"

namespace setu::planner::passes {

using setu::planner::topo::Link;

cir::Program ShortestPathRouting::Run(const cir::Program& program) {
  auto rw = cir::ProgramRewriter(program);
  for (std::size_t i = 0; i < program.NumOperations(); ++i) {
    const auto& op = program.Operations()[i];
    std::visit(
        [&](const auto& concrete) {
          using T = std::decay_t<decltype(concrete)>;
          if constexpr (std::is_same_v<T, cir::CopyOp>) {
            std::cout << program.GetValueInfo(concrete.src).ToString()
                      << std::endl;
            std::cout << program.GetValueInfo(concrete.dst_in).ToString()
                      << std::endl;
            auto src_val_info = program.GetValueInfo(concrete.src);
            auto dst_val_info = program.GetValueInfo(concrete.dst_in);
            // builder guarantees that num bytes is the same for src, dst values
            auto bytes = src_val_info.NumBytes();

            auto path_opt =
                topo_->ShortestPath(src_val_info.device, dst_val_info.device,
                                    [bytes](const Link& l) -> float {
                                      return l.TransferTimeUs(bytes);
                                    });
            ASSERT_VALID_RUNTIME(path_opt.has_value(),
                                 "No path exists between {} and {}",
                                 src_val_info.device, dst_val_info.device);

            auto path = path_opt.value();

            // shortest path is the same as direct transfer
            // no additional hops required
            if (path.hops.size() == 2) {
              rw.CloneOp(i);
              return;
            }

            auto num_elements = src_val_info.size_elements;
            auto dt = src_val_info.dtype;
            auto num_tmps = path.hops.size() - 2;
            std::vector<cir::Value> tmps;
            tmps.reserve(num_tmps);
            for (const auto& hop :
                 std::span(path.hops.begin() + 1, path.hops.end() - 1)) {
              tmps.emplace_back(
                  rw.Target().EmitAllocTmp(hop, num_elements, dt));
            };

            cir::Value prev = concrete.src;
            for (std::size_t j = 0; j < tmps.size(); ++j) {
              auto& tmp = tmps[j];
              auto tmp_out = rw.Target().EmitCopy(prev, tmp);
              // update tmp binding to enforce ordering
              tmp = tmp_out;
              prev = tmp_out;
            }
            auto dst_out = rw.Target().EmitCopy(prev, concrete.dst_in);
            rw.MapValue(concrete.dst_out, dst_out);
            return;
          }

          // fallthrough
          rw.CloneOp(i);
        },
        op.op);
  }
  return rw.Finish();
}
}  // namespace setu::planner::passes
