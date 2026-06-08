from pydantic import BaseModel, Field
from typing import List, Optional

from sqlalchemy import TIMESTAMP

class OrderSchema(BaseModel):
    order_id: str = Field(..., description="The ID of the order")
    customer_id: str = Field(..., description="The ID of the customer")
    order_status: str = Field(..., description="The status of the order")
    order_purchase_timestamp: TIMESTAMP = Field(..., description="The timestamp when the order was purchased")
    order_approved_at: Optional[TIMESTAMP] = Field(None, description="The timestamp when the order was approved")
    order_delivered_carrier_date: Optional[TIMESTAMP] = Field(None, description="The timestamp when the order was delivered by the carrier")
    order_delivered_customer_date: Optional[TIMESTAMP] = Field(None, description="The timestamp when the order was delivered to the customer")
    order_estimated_delivery_date: Optional[TIMESTAMP] = Field(None, description="The estimated timestamp when the order will be delivered")