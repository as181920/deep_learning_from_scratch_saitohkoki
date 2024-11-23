require "torch-rb"

module Global
  # DEVICE = "cpu".freeze
  DEVICE = Torch::CUDA.available? ? "cuda" : "cpu"
end
