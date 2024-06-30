const Messages = require("../models/messageModel");
const mongoose = require('mongoose');
const axios = require('axios');

module.exports.getMessages = async (req, res, next) => {
  try {
    const {conversationId} = req.body;
    const id = new mongoose.Types.ObjectId(conversationId);
    const messages = await Messages.find({
        conversationId: id,
      });

    messages.sort((a, b) => {
    const timestampA = new Date(a.timestamp);
    const timestampB = new Date(b.timestamp);
    
    return timestampA - timestampB;
    });
    return res.json(messages);
  } catch (ex) {
    next(ex);
  }
};

module.exports.addMessage = async (req, res, next) => {
    const {message, conversationId, sender} = req.body;
    if(message == undefined || conversationId == undefined || sender == undefined || 
      !message || !conversationId || !sender
    )
      return res.status(400).json({msg: 'Field validation error.'})

    const id = new mongoose.Types.ObjectId(conversationId);
    try {
      const newMessage = new Messages({
        message, 
        conversationId: id,
        sender
      });

      const response = await newMessage.save();
      return res.json(response);
    } catch (ex) {
      next(ex);
    }
  };

  module.exports.processData = async (req, res, next) => {
    const {text, videoFrame} = req.body;
    if(text == undefined || videoFrame == undefined)
      return res.status(400).json({msg: 'Field validation error.'})

    const response = await axios.post(process.env.SENTIMENT_ANALYSIS_ROUTE, {
      text: text,
      videoFrame: videoFrame
    });

    const { image_sentiment, text_sentiment } = response.data;

    if(image_sentiment == undefined || text_sentiment == undefined)
      return res.status(400).json({msg: 'Sentiment analysis could not be processed.'})

    return res.json(response.data);
  };