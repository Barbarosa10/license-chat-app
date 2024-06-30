const User = require("../models/userModel");
const Conversation = require("../models/conversationModel");

module.exports.getAllContacts = async (req, res, next) => {
  try {
    const users = await User.find({ _id: { $ne: req.params.id } }).select([
      "email",
      "username",
      "avatarImage",
      "_id",
    ]);
    return res.json(users);
  } catch (ex) {
    next(ex);
  }
};

module.exports.getAllConversations = async (req, res, next) => {
  try {
    const conversations = await Conversation.find({ 
      participants: { $in: [req.params.username] } 
    }).select([
      "participants",
      "lastMessage",
      "_id",
      "timestamp"
    ]);
    return res.json(conversations);
  } catch (ex) {
    next(ex);
  }
};

module.exports.getContact = async (req, res, next) => {
  try {
    const user = await User.find({ username: req.params.username  }).select([
      "avatarImage",
      "_id",
      "username"
    ]);

    return res.json(user);
  } catch (ex) {
    next(ex);
  }
};

module.exports.createConversation = async (req, res, next) => {
  const {participants} = req.body;
  const lastMessage = "";
  try {
    const newConversation = new Conversation({
      participants,
      lastMessage
    });

    const savedConversation = await newConversation.save();
    return res.json(savedConversation);
  } catch (ex) {
    next(ex);
  }
};

module.exports.updateLastMessage = async (req, res, next) => {
  const {lastMessage, conversationId} = req.body;

  try {
    const updatedLastMessage = {
      $set: {
        lastMessage,
        timestamp: Date.now(),
      },
    };

    const result = await Conversation.updateOne({_id: conversationId}, updatedLastMessage)
    return res.json(result);
  } catch (ex) {
    next(ex);
  }
};