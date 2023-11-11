const mongoose = require("mongoose");

const messageSchema = mongoose.Schema(
  {
    message: {
        type: String,
         required: true,
    },
    
    conversationId: {
        type: mongoose.Schema.Types.ObjectId,
        required: true,
    },

    sender: {
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

module.exports = mongoose.model("messages", messageSchema);
