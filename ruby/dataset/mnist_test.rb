require_relative "../test/test_helper"
require_relative "mnist"

describe Mnist do
  it "should download files" do
    Mnist.load

    assert_path_exists File.expand_path("t10k-labels-idx1-ubyte.gz", __dir__)

    assert_equal \
      Mnist::FILE_CHECKSUMS["train-images-idx3-ubyte.gz"],
      Digest::MD5.file(File.expand_path("train-images-idx3-ubyte.gz", __dir__)).to_s
  end
end
