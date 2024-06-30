const mongoose = require("mongoose");

const conversationSchema = mongoose.Schema(
  {
    participants: {
      type: Array,
      required: true,
      min: 2,
      max:2
    },

    lastMessage: {
      type: String,
      default: "",
    },
    
    timestamp: {
      type: Date,
      default: Date.now()
    }
  }
);

module.exports = mongoose.model("conversations", conversationSchema);