const mongoose = require("mongoose");

const conversationSchema = mongoose.Schema(
  {
    participants: {
      type: Array,
      ref: "users",
      required: true,
      min: 2,
      max:2
    },

    lastMessage: {
      type: String
    }
  }
);

module.exports = mongoose.model("conversations", conversationSchema);
