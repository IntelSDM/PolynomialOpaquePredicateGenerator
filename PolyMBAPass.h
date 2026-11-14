#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

struct PolyMBAPass : llvm::PassInfoMixin<PolyMBAPass>
{
    llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);
};

bool RunPolyMBA(llvm::Module& m);

extern llvm::cl::opt<bool> PolyMBAEnable;