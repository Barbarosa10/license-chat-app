const mongoose = require("mongoose");

const conversationSchema = mongoose.Schema(
  {
    sender: {
        type: String,
        ref: "users",
        required: true,
    },

    receiver: {
        type: String,
        ref: "users",
        required: true,
    },

    timestamp: {
        type: String,
        required: true,
    }
  }
);

module.exports = mongoose.model("conversations", conversationSchema);
