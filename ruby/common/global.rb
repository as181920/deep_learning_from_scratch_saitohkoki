require "torch-rb"
require "debug"

module Global
  DEVICE = Torch::CUDA.available? ? ENV.fetch("DEVICE", "cuda") : "cpu"
end
