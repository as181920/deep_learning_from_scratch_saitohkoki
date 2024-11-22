require "torch-rb"

module Loss
  module_function

  DELTA = Torch.tensor(10**-7)

  def mean_square_error(y, t)
    ((y - t)**2).sum / 2
  end

  def cross_entropy_error_with_one_hot(y, t)
    -(t * Torch.log(y + DELTA)).sum
  end

  def cross_entropy_error_with_mini_batch_one_hot(y, t)
    if y.ndim == 1
      y = y.reshape(1, -1)
      t = t.reshape(1, -1)
    end

    batch_size = y.shape[0]
    -(t * Torch.log(y + DELTA)).sum / batch_size
  end

  def cross_entropy_error_with_mini_batch_index(y, t)
    if y.ndim == 1
      y = y.reshape(1, -1)
      t = t.reshape(1, -1)
    end

    batch_size = y.shape[0]
    -Torch.log(y[Array(0..batch_size.pred), t] + DELTA).sum / batch_size
  end
  alias_method :cross_entropy_error, :cross_entropy_error_with_mini_batch_index
  module_function :cross_entropy_error_with_mini_batch_index, :cross_entropy_error # rubocop:disable Style/AccessModifierDeclarations
end
