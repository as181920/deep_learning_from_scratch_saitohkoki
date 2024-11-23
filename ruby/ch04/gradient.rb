require "torch-rb"

module Gradient
  module_function

  STEP_SIZE = Torch.tensor(10.0**-4, dtype: :float64)

  # use lambda as function
  def numerical_diff(f, x)
    (f.call(x + STEP_SIZE) - f.call(x - STEP_SIZE)) / (STEP_SIZE * 2)
  end

  # x = Torch.tensor([x0, x1, x2, ...])
  def numerical_gradient(f, x) # rubocop:disable Metrics/MethodLength
    grad = Torch.zeros(x.shape)

    x.each_with_index do |value, index|
      original_value = value.clone

      x[index] = original_value + STEP_SIZE
      val_r = f.call(x)

      x[index] = original_value - STEP_SIZE
      val_l = f.call(x)

      x[index] = original_value

      grad[index] = (val_r - val_l) / (STEP_SIZE * 2)
    end

    grad
  end
end
