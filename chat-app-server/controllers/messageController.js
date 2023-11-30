const Messages = require("../models/messageModel");
const mongoose = require('mongoose');


module.exports.getMessages = async (req, res, next) => {
  try {
    const {conversationId} = req.body;
    const id = new mongoose.Types.ObjectId(conversationId);
    // console.log(id);
    const messages = await Messages.find({
        conversationId: id,
      });

    messages.sort((a, b) => {
    // Convert the timestamp strings to Date objects for comparison
    const timestampA = new Date(a.timestamp);
    const timestampB = new Date(b.timestamp);
    
    // Compare timestamps and return the result
    return timestampA - timestampB;
    });
    // console.log(messages);
    return res.json(messages);
  } catch (ex) {
    next(ex);
  }
};

module.exports.addMessage = async (req, res, next) => {
    const {message, conversationId, sender} = req.body;
    const id = new mongoose.Types.ObjectId(conversationId);
    try {
      const newMessage = new Messages({
        message, 
        conversationId: id,
        sender
      });
      // console.log(newConversation);
      const response = await newMessage.save();
      return res.json(response);
    } catch (ex) {
      next(ex);
    }
  };


