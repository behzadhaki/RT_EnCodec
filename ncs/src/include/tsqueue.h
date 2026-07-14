#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
template<typename T>class tsqueue{
std::mutex m_;std::deque<T>q_;std::condition_variable cv_;std::atomic<bool>sd_{false};
public:
~tsqueue(){shutdown();}
void enqueue(T v){std::lock_guard<std::mutex>l(m_);q_.push_back(std::move(v));cv_.notify_one();}
bool try_dequeue(T&o){std::lock_guard<std::mutex>l(m_);if(q_.empty())return false;o=std::move(q_.front());q_.pop_front();return true;}
bool wait_dequeue(T&o,int t=100){std::unique_lock<std::mutex>l(m_);if(!cv_.wait_for(l,std::chrono::milliseconds(t),[this]{return!q_.empty()||sd_;}))return false;if(q_.empty())return false;o=std::move(q_.front());q_.pop_front();return true;}
void clear(){std::lock_guard<std::mutex>l(m_);q_.clear();}
size_t size(){std::lock_guard<std::mutex>l(m_);return q_.size();}
void shutdown(){sd_=true;cv_.notify_all();}
};