require_relative "../test/test_helper"
require_relative "activation"

describe Activation do
  it "should implement step function" do
    assert_equal 0, Activation.step_function_for_num(-1)
    assert_equal 1, Activation.step_function_for_num(999)
  end

  it "should implement step function for tensor" do
    assert Torch.equal(
      Torch.tensor([0, 0, 0, 1, 1]),
      Activation.step_function(Torch.tensor([-99, -1, 0, 1, 999]))
    )
  end

  it "should implement sigmoid function" do
    x = Torch.tensor([-99, -2, 0, 1, 123])

    assert Torch.equal(Torch.sigmoid(x), Activation.sigmoid(x))
  end

  it "should implement relu function" do
    x = Torch.tensor([-99, -2, 0, 1, 123])

    assert Torch.equal(Torch.tensor([0, 0, 0, 1, 123]), Activation.relu(x))
  end

  it "should implement identity function" do
    x = Torch.tensor([-99, -2, 0, 1, 1234])

    assert Torch.equal(x, Activation.identity_function(x))
  end
end
